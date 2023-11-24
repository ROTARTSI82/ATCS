import torch
import chess
import math

from torch.nn import Linear

import functools, random

dtype = torch.float32
device = torch.device('cuda')
chunk = 960 * 128 * 2


def pos_encode(val, num, val_max):
    # bug: i/2 should be i//2 but too late! i've already trained around that
    return [val / val_max] + [math.sin(math.pi * val / 2**(i/2)) for i in range(num*2)]


def preprocess(pos: chess.Board):
    ret = []
    move_enc = []
    is_white = pos.turn == chess.WHITE
    for rank in range(8):
        for file in range(8):
            # flip the board for point of view change
            sq = chess.square(file, rank if is_white else 7 - rank)
            pc = pos.piece_at(sq)

            pos_enc = pos_encode(rank, 3, 8) + pos_encode(file, 3, 8)
            encoding = [int(is_white)] + pos_enc

            encoding += [float(pc is not None and pc.color == (color ^ is_white)) for color in chess.COLORS]
            encoding += [float(pc is not None and pc.piece_type == piece_type) for piece_type in chess.PIECE_TYPES]
            encoding += [float(pos.ep_square == sq)]
            encoding += [float(side(is_white ^ color)) for color in chess.COLORS
                         for side in (pos.has_queenside_castling_rights, pos.has_kingside_castling_rights)]

            # possibly remove these, they probably dont contribute much
            encoding += pos_encode(pos.halfmove_clock, 8, 100)
            encoding += pos_encode(pos.fullmove_number / 16, 4, 6)
            ret.append(encoding)

    for move in pos.legal_moves:
        coords = [func(sq) for func in (chess.square_rank, chess.square_file)
                  for sq in (move.from_square, move.to_square)]

        encoding = [int(is_white)]
        for c in coords:
            encoding += pos_encode(c, 3, 8)

        # additional stuff like check, drop, and promotion are probably helpful here
        move_enc.append(encoding)

    return torch.tensor(ret, device=device, dtype=dtype), torch.tensor(move_enc, device=device, dtype=dtype)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_in = torch.nn.Sequential(Linear(54, 512, bias=False, dtype=dtype), torch.nn.GELU())
        self.moves_in = torch.nn.Sequential(Linear(29, 512, bias=False, dtype=dtype), torch.nn.GELU())
        self.trans = torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=8, num_decoder_layers=8,
                                          dim_feedforward=2048, bias=True, activation=torch.nn.GELU(),
                                          norm_first=True, batch_first=True, dropout=0.1, dtype=dtype)
        self.final_output = Linear(512, 1, bias=False, dtype=dtype)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, encoder, decoder):
        to_dec = self.moves_in(decoder)
        to_enc = self.pos_in(encoder)
        transformer = self.trans(to_enc, to_dec)
        return self.softmax(self.final_output(transformer).view(-1))


print("pytorch loaded - constructing model")
model = Model().to(device).train(False)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, eps=1e-5, betas=(0.9, 0.95), weight_decay=0.1)
# sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, gamma=0.975)
loss_fn = torch.nn.CrossEntropyLoss()

dim = sum([functools.reduce(lambda a, b: a * b, list(parameter.shape)) for parameter in model.parameters()])
print(f"dim={dim}")

# last_case = 0
# dicts = torch.load('modelv2.txt')
# model.load_state_dict(dicts['model'])
# # opt.load_state_dict(dicts['optim'])
# # sched.load_state_dict(dicts['sched'])
# last_case = dicts['last_case']
# del dicts
print('constructed model')

print(__name__)
if __name__ == "__main__":
    model = model.train(True)
    loss_history = [0]

    print('reading dataset')
    with open("/home/shared/chess/lichess_db_puzzle.csv", "r") as fp:
        lines = [line.split(',') for line in fp.read().split('\n')[1:] if line]
        random.shuffle(lines)
        lines = sorted(lines, key=lambda x: int(x[3]) - (1000 if "opening" in x[7] else 0) - (200 if "advantage" in x[7] else 0) + (100 if "mate" in x[7] else 0))
    print('dataset read')

    to_train = lines[last_case:last_case+chunk]
    tot = len(to_train)
    board = chess.Board()

    tot_pass = 0
    tot_seen = 0

    for case, puzzle in enumerate(to_train):
        moves = [chess.Move.from_uci(m) for m in puzzle[2].split(' ')]
        print(f"{case}/{tot}\t{100*case/tot:.2f}%\t{loss_history[-1]:.5f}\t{puzzle[3]}\t{puzzle}")
        board.set_fen(puzzle[1])
        passed = 0

        for best in moves:
            for idx, legal in enumerate(board.legal_moves):
                if legal == best:
                    enc, mov_enc = preprocess(board)

                    opt.zero_grad()
                    out = model(enc, mov_enc)
                    passed += int(torch.argmax(out) == idx)

                    expected = torch.zeros(list(out.shape)[0], device=device, dtype=dtype)
                    expected[idx] = 1

                    loss = loss_fn(out, expected)
                    # loss = -torch.log(out[idx])
                    loss_history.append(float(loss))
                    loss.backward()
                    opt.step()
                    break
            else:
                print("illegal move?")
            board.push(best)

        tot_pass += passed
        tot_seen += len(moves)
        print(f"passed {passed}/{len(moves)} = {100*passed/(len(moves)):.2f}%\t{100*tot_pass/tot_seen}")

        if case % 128 == 0:
            print(f"\n============\ttotal = {100*tot_pass/tot_seen}\t============\n")
            tot_pass = tot_seen = 0
            # sched.step()

    # torch.save({'model': model.state_dict(),
    #             'optim': opt.state_dict(),  # 'sched': sched.state_dict(),
    #             'last_case': last_case + chunk}, 'modelv2.txt')
    with open("loss.csv", 'w') as fp:
        fp.write(",".join([str(i) for i in loss_history]))
