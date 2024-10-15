import msvcrt
import random
import shutil
import sys
import time
from enum import StrEnum
from typing import Literal, NamedTuple

up_and_left: str = "\033[F"
up_one: str = "\033[A"

type Cell = offset | Literal[1, 0]
type Row = list[Cell]
type Offset = Literal[-1, 0, 1]


class offset(NamedTuple):
    y_offset: Offset
    x_offset: Offset

    def __neg__(self) -> "offset":
        return offset(y_offset=self.y_offset * -1, x_offset=self.x_offset * -1)

    def __repr__(self) -> str:
        return f"0(y={self.y_offset}, x={self.x_offset})"


class coord(NamedTuple):
    y_index: int
    x_index: int

    def __add__(self, obj: object) -> "coord":
        if not isinstance(obj, offset):
            raise ValueError(
                f"Can only add object of type {offset.__name__!r} to {coord.__name__!r}"
            )

        return coord(
            y_index=self.y_index + obj.y_offset, x_index=self.x_index + obj.x_offset
        )

    def __repr__(self) -> str:
        return f"c(y={self.y_index}, x={self.x_index})"


class BoardDetails(NamedTuple):
    board_height: int
    board_width: int

    left_start: int
    right_last: int

    top_start: int
    bottom_last: int


class ANSICodes(StrEnum):
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    CLEAR_SCREEN = "\033[2J"
    CLEAR_SCREEN_N3 = "\033[3J"
    MOVE_CURSOR = "\033[{y};{x}H"


class Dead(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class _BoardList:
    def __init__(self, height: int, width: int) -> None:
        self.__height: int = height
        self.__width: int = width

        self.__board: list[Row] = [[0 for _ in range(width)] for _ in range(height)]

    def __getitem__(self, key: coord) -> Cell:
        return self.__board[key.y_index][key.x_index]

    def __setitem__(self, key: coord, value: Cell) -> None:
        self.__board[key.y_index][key.x_index] = value

    def test_coord(self, co: coord) -> bool:
        return not (
            co.y_index < 0
            or co.y_index >= self.__height
            or co.x_index < 0
            or co.x_index >= self.__width
        )


class _EndChars:
    __end_char_dict = {
        "head": {
            offset(0, 1): "|>",
            offset(0, -1): "<|",
            offset(1, 0): "\\/",
            offset(-1, 0): "/\\",
        },
        "tail": {
            offset(0, 1): "=>",
            offset(0, -1): "<=",
            offset(1, 0): "||",
            offset(-1, 0): "||",
        },
    }

    @classmethod
    def get_head(cls, key: offset) -> str:
        return _EndChars.__end_char_dict["head"][key]

    @classmethod
    def get_tail(cls, key: offset) -> str:
        return _EndChars.__end_char_dict["tail"][key]

    # @classmethod
    # def get_body(cls, key: offset) -> str:
    #     return _EndChars.__end_char_dict["tail"][key]


class Board:
    @classmethod
    def create_board_details(cls, margin: int) -> BoardDetails:
        term_size = shutil.get_terminal_size()

        height = term_size.lines
        width = term_size.columns

        board_height = height - margin * 2
        board_width = width - margin * 2

        b_odd = board_width % 2

        board_width -= b_odd

        left_start = margin + 1
        right_last = width - margin - b_odd
        top_start = margin + 1
        bottom_last = height - margin

        return BoardDetails(
            board_height=board_height,
            board_width=board_width // 2,
            left_start=left_start,
            right_last=right_last,
            top_start=top_start,
            bottom_last=bottom_last,
        )

    def __init__(self, details: BoardDetails) -> None:
        self.__dir: offset = offset(0, 1)

        self.__details: BoardDetails = details

        self.__board: _BoardList = _BoardList(
            height=details.board_height,
            width=details.board_width,
        )
        self.__update_cells: set[coord] = set()

        self.__index_head: coord = coord(0, 2)
        self.__index_tail: coord = coord(0, 0)

        self.__points: int = 0
        self.__moves: int = 0

        self.__char_buffer: list[bytes] = []

        self.__init_board()

    @property
    def points(self) -> int:
        return self.__points

    def __init_board(self) -> None:
        self.__board[coord(0, 0)] = offset(0, 1)
        self.__board[coord(0, 1)] = offset(0, 1)
        self.__board[coord(0, 2)] = offset(0, 0)

        self.__update_cells.add(coord(0, 1))

    def __spawn_food(self) -> None:
        while True:
            new_food: coord = coord(
                random.randint(0, self.__details.board_height - 1),
                random.randint(0, self.__details.board_width - 1),
            )

            if self.__board[new_food] == 0:
                self.__board[new_food] = 1
                self.__update_cells.add(new_food)
                return None

    def __validate_next_head(self) -> bool:
        """Check that the next board state is valid. If invalid, a `Dead` exception is raised.
        If the head hits food, return True, else False."""

        next_head = self.__index_head + self.__dir

        if not self.__board.test_coord(next_head):
            raise Dead("Hit Wall")

        next_cell_contents: Cell = self.__board[next_head]

        if next_cell_contents == 1:
            return True
        elif next_cell_contents == 0:
            return False

        raise Dead("Hit Self")

    def __update(self) -> None:
        acquired_food: bool = self.__validate_next_head()
        self.__moves += 1

        self.__board[self.__index_head] = self.__dir
        self.__update_cells.add(self.__index_head)

        self.__index_head += self.__dir
        self.__board[self.__index_head] = offset(0, 0)

        if acquired_food:
            self.__points += 1
            self.__spawn_food()
            return None

        self.__update_cells.add(self.__index_tail)

        tail_offset = self.__board[self.__index_tail]

        self.__board[self.__index_tail] = 0
        self.__index_tail += tail_offset

    def __print_board(self) -> None:
        # "░░" or "██"
        # body: str = (
        #     "abcdefghijklmnopqrstuvwxyz"
        #     + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        #     + "1234567890"
        #     + "!@#$%^&*"
        # )

        apple: str = "()"
        empty: str = "  "
        body: str = "░░"

        tail_offset = self.__board[self.__index_tail]
        assert isinstance(tail_offset, offset)

        head: str = _EndChars.get_head(self.__dir)
        tail: str = f"[]"  # _EndChars.get_tail(-tail_offset)

        moves_str: str = f"Moves: {self.__moves}"

        update_str: str = (
            ANSICodes.MOVE_CURSOR.format(
                y=self.__details.top_start - 1,
                x=self.__details.left_start,
            )
            + f"Score: {self.__points}"
            + ANSICodes.MOVE_CURSOR.format(
                y=self.__details.top_start - 1,
                x=self.__details.right_last - len(moves_str) + 1,
            )
            + moves_str
            + ANSICodes.MOVE_CURSOR.format(
                y=self.__details.top_start + self.__index_head.y_index,
                x=self.__details.left_start + self.__index_head.x_index * 2,
            )
            + head
            + ANSICodes.MOVE_CURSOR.format(
                y=self.__details.top_start + self.__index_tail.y_index,
                x=self.__details.left_start + self.__index_tail.x_index * 2,
            )
            + tail
        )

        while self.__update_cells:
            co = self.__update_cells.pop()

            cell: Cell = self.__board[co]
            display_value: str

            if cell == 0:
                display_value = empty
            elif cell == 1:
                display_value = apple
            else:
                display_value = body

            update_str += (
                ANSICodes.MOVE_CURSOR.format(
                    y=self.__details.top_start + co.y_index,
                    x=self.__details.left_start + co.x_index * 2,
                )
                + display_value
            )

        print(update_str + ANSICodes.HIDE_CURSOR, end="", flush=True)

    def __read_input(self) -> None:
        max_chars: int = 5

        read_chars: int = 0
        chars: list[bytes] = []
        while msvcrt.kbhit() and read_chars < max_chars:
            read_chars += 1
            chars.append(msvcrt.getch())

        if chars:
            self.__char_buffer = chars

    def __parse_input(self) -> None:
        prev_char: bytes | None = None
        move_type: offset | None = None

        while self.__char_buffer:
            c = self.__char_buffer.pop(0)

            if c in b"wasdWASD" or (prev_char == b"\xe0" and c in b"HPMK"):
                prev_char = None
            else:
                prev_char = c
                continue

            match c:
                case b"w" | b"W" | b"H":
                    move_type = offset(-1, 0)
                case b"s" | b"S" | b"P":
                    move_type = offset(1, 0)
                case b"d" | b"D" | b"M":
                    move_type = offset(0, 1)
                case b"a" | b"A" | b"K":
                    move_type = offset(0, -1)
                case _:
                    raise ValueError("Wtf?")

            if (
                move_type.x_offset + self.__dir.x_offset
                or move_type.y_offset + self.__dir.y_offset
            ):
                self.__dir = move_type
            return None

    def start(self) -> None:
        self.__spawn_food()
        self.__print_board()

    def update(self, update_rate: float) -> None:
        self.__read_input()
        self.__parse_input()
        self.__update()
        self.__print_board()
        time.sleep(update_rate)


def _run_loop(board: Board) -> None:
    update_rate: float = 5 / 60

    while True:
        board.update(update_rate)


def _print_pattern(n: int, t: bool = False) -> list[str]:
    space = " "
    max_digits = len(str(n))

    output: list[str] = []
    for magnitude in range(max_digits - 1, 0, -1):
        row = "".join(
            [
                f"{(i%10 or space):<10}" * (10 ** (magnitude - 1))
                for i in range(n // (magnitude * 10) + 1)
            ]
        )[1 : n + 1]
        output.append(row)

    output.append(("123456789 " * (n // 10 + 1))[:n])

    if t:
        return list("".join(i) for i in zip(*output))

    return output


def _do_test() -> None:
    cols = shutil.get_terminal_size().columns
    rows = shutil.get_terminal_size().lines

    col_strs = _print_pattern(cols)
    row_strs = _print_pattern(rows, t=True)
    print(
        "\n".join(
            map(lambda x: " " * len(row_strs[0]) + x[len(row_strs[0]) :], col_strs)
        )
    )
    print("\n".join(row_strs[len(col_strs) :]))


def main(*args: str) -> int:
    print(ANSICodes.CLEAR_SCREEN_N3, end="", flush=True)

    details = Board.create_board_details(1)
    board = Board(details)

    board.start()
    msvcrt.getch()

    try:
        _run_loop(board)
    except Dead as ex:
        print(
            ANSICodes.MOVE_CURSOR.format(
                y=details.bottom_last + 1, x=details.left_start
            )
            + f"{ex.args[0]} - Score: {board.points}"
        )

    return 0


if __name__ == "__main__":
    try:

        main(*sys.argv)
    finally:
        print(ANSICodes.SHOW_CURSOR, end="")
    # try:
    #     main(*sys.argv)
    # except Exception as ex:
    #     print("|" + " " * 5 + str(hash(ex)) + " " * 5 + str(ex) + " " * 5 + "|")

# print(
#     ANSICodes.MOVE_CURSOR.format(y=details.min_y_offset, x=details.min_x_offset)
#     + "."
#     + ANSICodes.MOVE_CURSOR.format(
#         y=details.min_y_offset,
#         x=details.terminal_width - details.min_x_offset,
#     )
#     + ".",
#     end="",
#     flush=True,
# )

# print(
#     ANSICodes.MOVE_CURSOR.format(y=details.board_y_offset, x=details.board_x_offset)
#     + "*"
#     + ANSICodes.MOVE_CURSOR.format(
#         y=details.board_y_offset,
#         x=details.terminal_width - details.board_x_offset,
#     )
#     + "*",
#     end="",
#     flush=True,
# )

# print(
#     ANSICodes.MOVE_CURSOR.format(
#         y=details.terminal_height - details.min_y_offset,
#         x=details.min_x_offset,
#     )
#     + "."
#     + ANSICodes.MOVE_CURSOR.format(
#         y=details.terminal_height - details.min_y_offset,
#         x=details.terminal_width - details.min_x_offset,
#     )
#     + ".",
#     end="",
#     flush=True,
# )
# print(
#     ANSICodes.MOVE_CURSOR.format(
#         y=details.terminal_height - details.board_y_offset,
#         x=details.board_x_offset,
#     )
#     + "*"
#     + ANSICodes.MOVE_CURSOR.format(
#         y=details.terminal_height - details.board_y_offset,
#         x=details.terminal_width - details.board_x_offset,
#     )
#     + "*",
#     end="",
#     flush=True,
# )

# print(
#     ANSICodes.MOVE_CURSOR.format(
#         y=details.terminal_height, x=details.terminal_height
#     ),
#     flush=True,
# )

# print("." * shutil.get_terminal_size().columns)
# # print("\n".join(map(str, range(1, shutil.get_terminal_size().lines))))
# return 0
