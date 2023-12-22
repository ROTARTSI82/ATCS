import torch
import chess
import chess.pgn
from custom_transform import *

import functools, random


def preprocess(pos: chess.Board, moves):
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

    for move in moves:
        coords = [func(sq) for func in (chess.square_rank, chess.square_file)
                  for sq in (move.from_square, move.to_square)]

        encoding = [int(is_white)]
        for c in coords:
            encoding += pos_encode(c, 3, 8)

        # additional stuff like check, drop, and promotion are probably helpful here
        move_enc.append(encoding)

    return torch.tensor(ret, device=device, dtype=dtype), torch.tensor(move_enc, device=device, dtype=dtype)


class ChessModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = torch.nn.Parameter(torch.randn(64, d_model))
        self.mov_src = torch.nn.Embedding(64, d_model)
        self.mov_dst = torch.nn.Embedding(64, d_model)
        self.piece_type = torch.nn.Embedding(13, d_model)  # 6-pieces per side plus 1 empty
        self.fiftymove = torch.nn.Embedding(75, d_model)
        print(self.fiftymove.weight.shape)
        self.fiftymove.weight = torch.zeros(75, d_model)

        self.encoder = torch.nn.ModuleList([Attention() for _ in range(n_enc)])
        self.decoder = torch.nn.ModuleList([Attention() for _ in range(n_dec * 2)])  # cross-attention

        self.encoder_mlp = torch.nn.ModuleList([MLP() for _ in range(n_enc)])
        self.decoder_mlp = torch.nn.ModuleList([MLP() for _ in range(n_dec)])

    def forward(self, pieces, mov1, mov2):
        moves = self.mov_src(mov1) + self.mov_dst(mov2)
        return None


print("pytorch loaded - constructing model")
model = Model().to(device).train(True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, eps=1e-5, betas=(0.9, 0.95), weight_decay=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

dim = sum([functools.reduce(lambda a, b: a * b, list(parameter.shape)) for parameter in model.parameters()])
print(f"dim={dim}")

model.load_state_dict(torch.load('bigcarlsen.txt'))
print(model)
print(type(model))

print(__name__)
if __name__ == "__main__":

    print('model loaded - reading dataset')
    # with open("/home/shared/chess/lichess_db_puzzle.csv", "r") as fp:
    #     lines = [line.split(',') for line in fp.read().split('\n')[1:] if line]
    #     random.shuffle(lines)
    #     lines = sorted(lines, key=lambda x: int(x[3]) - (1000 if "opening" in x[7] else 0) - (200 if "advantage" in x[7] else 0) + (100 if "mate" in x[7] else 0))
    games = []
    with open("/home/shared/chess/Carlsen.pgn" if False else "/home/shared/chess/lichess_db_standard_rated_2016-02.pgn", 'r') as fp:
        for i in range(chunk):
            g = chess.pgn.read_game(fp)
            if g is None:
                break
            games.append(g)
    random.shuffle(games)
    games = chunk*[games[0]]
    print(games[0])
    print('dataset read')

    tot = len(games)
    board = chess.Board()

    tot_pass = 0
    tot_seen = 0
    loss_history = [0]
    for case, game in enumerate(games):
        # moves = [chess.Move.from_uci(m) for m in puzzle[2].split(' ')]
        print(f"{case}/{tot}\t{100*case/tot:.2f}%\t{loss_history[-1]:.5f}")
        board = game.board()
        passed = 0
        tot_mov = 0
        opt.zero_grad()
        loss_accum = None
        for best in game.mainline_moves():
            tot_mov += 1
            movs = list(board.legal_moves)
            for idx, legal in enumerate(movs):
                if legal == best:
                    # opt.zero_grad()  # !!

                    enc, mov_enc = preprocess(board, movs)
                    out = model(enc, mov_enc)
                    passed += int(torch.argmax(out) == idx)

                    expected = torch.zeros(list(out.shape)[0], device=device, dtype=dtype)
                    expected[idx] = 1

                    iloss = loss_fn(out, expected)
                    # iloss = -torch.log(out[idx])
                    if loss_accum is None:
                        loss_accum = iloss
                        # loss_accum.backward()
                        # opt.step()
                        # loss_history.append(float(loss_accum))
                        # loss_accum = None
                    else:
                        loss_accum += iloss
                    break
            else:
                print("illegal move?")
            board.push(best)

        if tot_mov == 0:
            print("empty game")
            continue

        loss_accum /= float(tot_mov)
        loss_history.append(float(loss_accum))
        loss_accum.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        tot_pass += passed
        tot_seen += tot_mov
        print(f"passed {passed}/{tot_mov} = {100*passed/tot_mov:.2f}%\t{100*tot_pass/tot_seen}")

        if case % 128 == 0:
            print(f"\n============\ttotal = {100*tot_pass/tot_seen}\t============\n")
            tot_pass = tot_seen = 0
            # sched.step()

    torch.save(model.state_dict(), 'bigcarlsen.txt')
    #print(model.state_dict())
    #print([i.grad for i in model.parameters()])
    # model_raw.eval()
    # torch.save(model_raw.state_dict(), 'test.txt')
    # torch.jit.script(model_raw).save('small.pt')  # Save
    with open("loss2.csv", 'w') as fp:
        fp.write(",".join([str(i) for i in loss_history]))
