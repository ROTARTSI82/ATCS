import chess
import chess.pgn
from custom_transform import *
import random
from torch import nn

chunk = 1


def preprocess(pos: chess.Board, moves):
    pieces = []
    mov1 = []
    mov2 = []
    promo = []
    castleflags = [float(func(pos.turn ^ side)) for side in chess.COLORS for func in
                   (pos.has_queenside_castling_rights, pos.has_kingside_castling_rights)]
    halfmoves = [pos.halfmove_clock / 25]
    for rank in range(8):
        for file in range(8):
            # flip the board for point of view change
            sq = chess.square(file, rank if pos.turn else 7 - rank)
            pc = pos.piece_at(sq)

            if pos.ep_square == sq:
                pieces.append(1)
                continue
            if pc is None:
                pieces.append(0)
                continue
            pieces.append((1 if pc.color == pos.turn else 1+6) + pc.piece_type)

    for move in moves:
        promo.append(move.promotion - 1 if move.promotion is not None else 0)
        mov1.append(move.from_square)
        mov2.append(move.to_square)

    return (torch.tensor([t], device=device) for t in (pieces, mov1, mov2, promo, castleflags, halfmoves))


class ChessModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(64, d_model))
        self.piece_type = nn.Embedding(14, d_model)  # 6 pieces * 2 sides + 1 en passant + 1 empty

        self.mov_src = nn.Embedding(64, d_model)
        self.mov_dst = nn.Embedding(64, d_model)
        self.promote = nn.Embedding(5, d_model)  # 4 piece types + 1 none

        self.fiftymove = nn.Linear(1, d_model, bias=False)
        self.fiftymove.weight.data.fill_(0)

        self.castling = nn.Linear(4, d_model, bias=False)
        self.castling.weight.data.fill_(0)

        self.transformer = Transformer(cross_first=True)

        self.sel_proj = nn.Sequential(RMSNorm(d_model), nn.Linear(d_model, 1, bias=False))
        self.eval_proj = nn.Sequential(RMSNorm(d_model), nn.Linear(d_model, 4, bias=False))

        self.addup_query = nn.Parameter(torch.randn(1, 1, qk_size * n_head))
        self.addup = Attention(False, bias_out=True)
        self.addup_mlp = MLP()

    def forward(self, pieces, mov1, mov2, promo, castleflags, halfmoves):
        moves = self.mov_src(mov1) + self.mov_dst(mov2) + self.promote(promo)
        pc = self.pos_emb + self.piece_type(pieces) + self.fiftymove(halfmoves) + self.castling(castleflags)

        mov_base, ev_base = self.transformer(pc, moves)

        summed = self.addup(self.addup_query, ev_base)
        summed = summed + self.addup_mlp(summed)
        return (
            self.sel_proj(mov_base),
            self.eval_proj(summed)
        )


print("pytorch loaded - constructing model")
model = ChessModel().to(device).train(True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, eps=1e-5, betas=(0.9, 0.95), weight_decay=0.1)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
rmse = nn.MSELoss()

dim = sum(p.numel() for p in model.parameters())
print(f"Transformer with {dim/1000000} million params")

# load = torch.load('bigcarlsen.txt')
# model.load_state_dict(load['model'])
# print(model)
# print(type(model))

if __name__ == "__main__":

    print('model loaded - reading dataset')
    # with open("/home/shared/chess/lichess_db_puzzle.csv", "r") as fp:
    #     lines = [line.split(',') for line in fp.read().split('\n')[1:] if line]
    #     random.shuffle(lines)
    #     lines = sorted(lines, key=lambda x: int(x[3]) - (1000 if "opening" in x[7] else 0) - (200 if "advantage" in x[7] else 0) + (100 if "mate" in x[7] else 0))
    games = []
    with open("/home/shared/chess/Carlsen.pgn" if True else "/home/shared/chess/lichess_db_standard_rated_2016-02.pgn", 'r') as fp:
        for i in range(chunk):
            g = chess.pgn.read_game(fp)
            if g is None:
                break
            games.append(g)
    games = 32 * games
    random.shuffle(games)
    # games = chunk*[games[0]]
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
                    margs = preprocess(board, movs)
                    out = model(*margs)[0]
                    passed += int(torch.argmax(out) == idx)

                    expected = torch.tensor(idx, device=device)
                    # print(out.shape, expected, expected.shape)

                    iloss = loss_fn(out.view(-1), expected)
                    # iloss = -torch.log(out[idx])
                    if loss_accum is None:
                        loss_accum = iloss
                        # loss_accum.backward()
                        # opt.step()
                        # loss_history.append(float(loss_accum))
                        # loss_accum = None
                    else:
                        loss_accum = loss_accum + iloss
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

    torch.save({'model': model.state_dict()}, 'test.txt')
    #print(model.state_dict())
    #print([i.grad for i in model.parameters()])
    # model_raw.eval()
    # torch.save(model_raw.state_dict(), 'test.txt')
    # torch.jit.script(model_raw).save('small.pt')  # Save
    # with open("loss2.csv", 'w') as fp:
    #     fp.write(",".join([str(i) for i in loss_history]))
