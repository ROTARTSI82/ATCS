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

legals = list(board.legal_moves)
probs = model(*preprocess(board, legals))
bestmove = torch.argmax(probs)

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
                    legals = list(board.legal_moves)
                    probs = model(*preprocess(board, legals))
                    bestmove = torch.argmax(probs)
                    print(f"bestmove {legals[bestmove]}, raw = {probs}")
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

    for i, (move, tp) in enumerate(zip(legals, probs)):
        prob = float(tp)
        ival = int(255 * prob)

        start = (res[0] * chess.square_file(move.from_square) // 8 + res[0] // 16,
                 res[1] * (7 - chess.square_rank(move.from_square)) // 8 + res[1] // 16)

        end = (res[0] * chess.square_file(move.to_square) // 8 + res[0] // 16,
               res[1] * (7 - chess.square_rank(move.to_square)) // 8 + res[1] // 16)

        if i == bestmove:
            pygame.draw.line(screen, (255, 0, 0), start, end, 8)

    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()
