import chess
import chess.pgn
import torch.nn.init

from llm.custom_transform import *
import json
import math
from torch import nn

chunk = 1
conf = Config()


def preprocess(pos: chess.Board, moves):
    pieces = []
    mov1 = []
    mov2 = []
    promo = []
    castleflags = [float(func(pos.turn ^ flip)) for flip in [True, False] for func in
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

    def cond_flip(s):
        f = chess.square_file(s)
        r = chess.square_rank(s)
        return chess.square(f, r if pos.turn else 7 - r)

    for move in moves:
        promo.append(move.promotion - 1 if move.promotion is not None else 0)
        mov1.append(cond_flip(move.from_square))
        mov2.append(cond_flip(move.to_square))

    return [[t] for t in (pieces, mov1, mov2, promo, castleflags, halfmoves)]


class ChessModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(64, conf.d_model))
        self.piece_type = nn.Embedding(14, conf.d_model)  # 6 pieces * 2 sides + 1 en passant + 1 empty

        self.mov_src = nn.Embedding(64, conf.d_model)
        self.mov_dst = nn.Embedding(64, conf.d_model)
        self.promote = nn.Embedding(5, conf.d_model)  # 4 piece types + 1 none

        self.fiftymove = nn.Linear(1, conf.d_model, bias=False)
        self.fiftymove.weight.data.fill_(0)

        self.castling = nn.Linear(4, conf.d_model, bias=False)
        self.castling.weight.data.fill_(0)

        self.transformer = Transformer(conf, cross_first=True)

        self.sel_proj = nn.Sequential(RMSNorm(conf.d_model), nn.Linear(conf.d_model, 1, bias=False))

        # 3 for WDL, (2 for CP vs. Mate, 2 for CP score and mate score) * 2 for each side's advantage
        self.eval_proj = nn.Sequential(RMSNorm(conf.d_model), nn.Linear(conf.d_model, 11, bias=False))

        self.addup_query = nn.Parameter(torch.randn(1, 1, conf.qk_size * conf.n_head))
        self.addup = Attention(conf, False, bias_out=True)
        self.addup_mlp = MLP(conf)

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
no_smooth_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.0)

dim = sum(p.numel() for p in model.parameters())
print(f"Transformer with {dim/1000000} million params")

load = torch.load('eval_30M.bin')
model.load_state_dict(load['model'])
# print(model)
# print(type(model))

# code stolen from
# https://github.com/official-stockfish/Stockfish/blob/fbdf5d94a9a42acb92720a5896b16c92931ec3de/src/uci.cpp#L226
aes = (0.38036525, -2.82015070, 23.17882135, 307.36768407)
bs = (-2.29434733, 13.27689788, -14.26828904, 63.45318330)
pawn_normalize = sum(aes)


def win_rate_model(v, ply):
    m = min(240, ply) / 64.0

    a = (((aes[0] * m + aes[1]) * m + aes[2]) * m) + aes[3]
    b = (((bs[0] * m + bs[1]) * m + bs[2]) * m) + bs[3]

    x = min(max(v, -4000.0), 4000.0)
    return 1 / (1 + math.exp((a - x) / b))


def sgn(x):
    return 1 if x > 0 else -1


def rsqrt(x):
    return sgn(x) * math.sqrt(abs(x))


def usq(x):
    return sgn(x) * x**2


if __name__ == "__main__":
    dataset = open('/home/shared/chess/lichess_db_eval.json', 'r')
    dataset.seek(0, 2)
    tot = dataset.tell()
    dataset.seek(load['loc'])
    dataset.readline()

    board = chess.Board()

    batch = 0
    loss_hist = []
    while True:
        batch_size = 0
        cost = torch.tensor(0.0, device=device)
        for case in range(64):
            dat = json.loads(dataset.readline())
            board.set_fen(dat['fen'])
            legals = list(board.legal_moves)
            margs = preprocess(board, legals)

            ev = dat['evals'][0]['pvs'][0]
            best = chess.Move.from_uci(ev['line'].split(' ')[0])
            if board.piece_at(best.from_square).piece_type == chess.KING:
                # detect and correct castling. UCI encodes as moving king onto rook,
                # chess.py for some reason moves the king to the square where it actually ends up.
                if chess.square_distance(best.from_square, best.to_square) > 1:
                    best.to_square = best.from_square + (best.to_square - best.from_square + 1) // 2

            if best not in legals:
                print("illegal move!!!", best, legals)
                print(board)
                print(dat)
                continue
            true_mov = torch.tensor(legals.index(best), device=device)
            if 'cp' in ev:
                v = int(ev['cp']) * pawn_normalize / 100
            else:
                v = 32000 * max(min(int(ev['mate']), 1), -1)
            true_cat = torch.tensor(int('mate' in ev) + max(0, sgn(v)) * 2, device=device)
            win = win_rate_model(v, board.ply())
            loss = win_rate_model(-v, board.ply())
            draw = 1 - win - loss
            wdl = torch.tensor([win, draw, loss], device=device)
            true_score = rsqrt(float(ev['mate'])) if 'mate' in ev else rsqrt(float(ev['cp']) / 200)

            opt.zero_grad()
            fargs = [torch.tensor(t, device=device) for t in margs]
            m_sel, m_ev = model(*fargs)
            m_wdl = m_ev[0, 0, :3].view(-1)
            m_cat = m_ev[0, 0, 3:3+4].view(-1)
            m_score = m_ev[0, 0, true_cat + 7]

            cost += 2*loss_fn(m_sel.view(-1), true_mov)
            cost += loss_fn(m_cat, true_cat)
            cost += no_smooth_loss_fn(m_wdl, wdl)
            cost += (m_score - true_score)**2
            batch_size += 1

            if case == 63:
                sf = torch.softmax(m_wdl, dim=0)
                sf_cat = torch.softmax(m_cat, dim=0)
                scat = torch.argmax(sf_cat)
                sf_movs = torch.softmax(m_sel.view(-1), dim=0)
                mod_best = torch.argmax(sf_movs)
                factor = 2 if 'cp' in ev else 1
                print(f"{100 * dataset.tell() / tot:.2f}%\tloss={cost.item()/batch_size:.5f}\n", end='')
                print(f"\t\tPREDICT\tscore({torch.max(sf_cat):.2f}) = {usq(m_ev[0, 0, scat + 7]) * factor:.2f}", end='')
                print(f"\twdl {sf[0]:.3f} {sf[1]:.3f} {sf[2]:.3f}", end='')
                print(f"\tbestmove({torch.max(sf_movs):.3f}) {legals[mod_best.item()]}")

                print(f"\t\tTRUTH\tscore({sf_cat[true_cat]:.2f}) = {usq(true_score) * factor:.2f}", end='')
                print(f"\twdl {win:.3f} {draw:.3f} {loss:.3f}", end='')
                print(f"\tbestmove({sf_movs[true_mov.item()]:.3f}) {best}")

                print(f"\t\t{dat['fen']}")
                with open('loss.csv', 'a') as fp2:
                    fp2.write(f"{cost.item() / batch_size}, ")

        cost /= batch_size
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        batch += 1
        if batch % 42 == 0:
            print('=============== checkpoint saved ===================')
            torch.save({'model': model.state_dict(), 'loc': dataset.tell()}, 'eval_30M.bin')

model.eval()
