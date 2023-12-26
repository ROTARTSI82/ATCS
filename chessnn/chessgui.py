import pygame
from train import *

pygame.init()

board = chess.Board()

res = (640, 480)
screen = pygame.display.set_mode(res)
clock = pygame.time.Clock()
running = True

cursor = (0, 0)

selection = None


def load(x):
    return x, pygame.transform.scale(pygame.image.load(f"res/{x}.png").convert_alpha(), (res[0] // 8, res[1] // 8))


def to_sq(sel):
    files = "abcdefgh"
    return files[sel[0]] + str(1 + sel[1])


images = dict([load(('w' if color == chess.WHITE else 'b') + chess.piece_name(piece))
               for piece in chess.PIECE_TYPES for color in chess.COLORS])

legals = None
probs = None
bestmove = None


def mod():
    global legals, probs, bestmove
    legals = list(board.legal_moves)
    m_sel, m_ev = model(*[torch.tensor(t, device=device) for t in preprocess(board, legals)])
    m_wdl = m_ev[0, 0, :3].view(-1)
    m_cat = m_ev[0, 0, 3:3 + 4].view(-1)
    m_score = m_ev[0, 0, torch.argmax(m_cat) + 7]
    probs, wdl, cats, score = (
        torch.softmax(m_sel.view(-1), dim=0), torch.softmax(m_wdl, dim=0),
        torch.softmax(m_cat, dim=0), m_score.item()
    )
    bestmove = torch.argmax(probs)
    cat = torch.argmax(cats)
    print(f"bestmove({probs[bestmove]:.2f}) {legals[bestmove]}")
    print(f"\twdl {wdl[0]:.2f} {wdl[1]:.2f} {wdl[2]:.2f}")
    print(f"\t{['cp-', '#-', 'cp+', '#+'][cat]}({cats[cat]:.2f}) ", end='')
    print(f"{usq(score) * (2 if cat % 2 == 0 else 1):.2f}")


mod()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            new_select = (8 * event.pos[0] // res[0], 7 - 8 * event.pos[1] // res[1])
            if selection is None:
                selection = new_select
            elif new_select != selection:
                mov = chess.Move.from_uci(to_sq(selection) + to_sq(new_select))
                print(mov)
                if board.is_legal(mov):
                    board.push(mov)
                    mod()

                selection = None

    screen.fill("black")
    for rank in range(0, 8):
        for file in range(0, 8):
            rect = pygame.Rect(file*res[0]/8, (7-rank)*res[1]/8, res[0]/8, res[1]/8)
            piece = board.piece_at(chess.square(file, rank))
            col = (128, rank * 256 // 8, file * 256 // 8)
            pygame.draw.rect(screen, col, rect)
            if piece is not None:
                lookup = ('w' if piece.color == chess.WHITE else 'b') + chess.piece_name(piece.piece_type)
                screen.blit(images[lookup], rect)
            if selection == (file, rank):
                surf = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
                surf.fill((255, 255, 255, 64))
                screen.blit(surf, rect)

    for i, (move, tp) in enumerate(zip(legals, probs.view(-1).tolist())):
        # print(tp)
        prob = float(tp)
        ival = int(255 * prob)

        start = (res[0] * chess.square_file(move.from_square) // 8 + res[0] // 16,
                 res[1] * (7 - chess.square_rank(move.from_square)) // 8 + res[1] // 16)

        end = (res[0] * chess.square_file(move.to_square) // 8 + res[0] // 16,
               res[1] * (7 - chess.square_rank(move.to_square)) // 8 + res[1] // 16)

        if i == bestmove:
            pygame.draw.line(screen, (255, 0, 0), start, end, 16)
        else:
            pygame.draw.line(screen, (0, 0, ival), start, end, int(16 * max(prob, 1/16)))

    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()
