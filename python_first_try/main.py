from random import randrange

from curses_ui import GameUI
from game import Field


class UIMode:
    PRINT = 0
    CONSOLE_UI = 1


def main():
    ui_mode = UIMode.CONSOLE_UI
    era_num = 125
    field_size = 40

    field = Field(field_size)
    field.infect(randrange(0, field_size), randrange(0, field_size))

    if ui_mode == UIMode.PRINT:
        for i in range(era_num):
            field.show()
            field.change_the_era()
        field.show()
    elif ui_mode == UIMode.CONSOLE_UI:
        ui = GameUI(field, era_num, 50)
        ui.start_ui()
    else:
        print("Error: unknown UI mode!")


if __name__ == "__main__":
    main()
