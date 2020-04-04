# from curses import wrapper, delay_output, newwin
import curses
from abc import abstractmethod

from game import Field
from states import States, State, Statistics


class UIColors:
    NORMAL = 1
    INFECTED = 2
    PATIENT = 3
    DEAD = 4
    TEXT = 4


class BaseSubWindow:
    def __init__(self, stdscr, height, width, begin_y, begin_x):
        self.parent_screen = stdscr
        self.window = curses.newwin(height, width, begin_y, begin_x)

    @abstractmethod
    def render_window(self, *args, **kwargs):
        pass

    def window_formatting(self, *args, **kwargs):
        self.window.addstr(self.render_window(*args, **kwargs))

    def refresh(self, *args, **kwargs):
        self.window.clear()
        self.window_formatting(*args, **kwargs)
        self.window.refresh()


class ProgressBar(BaseSubWindow):
    def __init__(self, stdscr):
        super().__init__(stdscr, 1, stdscr.getmaxyx()[1],
                         stdscr.getmaxyx()[0] - 1, 0)

    def render_window(self, percent: float) -> str:
        work_w = self.window.getmaxyx()[1] - 7
        p = int(percent * 100)
        l = int(work_w * percent)
        r = work_w - l
        return '[' + ('#' * l) + (' ' * r) + f']{str(p).rjust(3)}%'


class FieldWindow(BaseSubWindow):
    def __init__(self, stdscr, field: Field):
        self.field: Field = field
        size: int = len(field.matrix)
        y: int = (stdscr.getmaxyx()[0] - size) // 2 - 1
        x: int = (stdscr.getmaxyx()[1] - size * 2) // 2 - 1
        super().__init__(stdscr, size + 2, size * 2 + 1, y, x)

    def render_window(self) -> (State, int, int):
        for y, line in enumerate(self.field.matrix):
            for x, elem in enumerate(line):
                yield elem.get_state(), y, x

    def window_formatting(self, *args, **kwargs):
        for state, y, x in self.render_window():
            if state == States.NORMAL:
                color = UIColors.NORMAL
            elif state == States.INFECTED:
                color = UIColors.INFECTED
            elif state == States.PATIENT:
                color = UIColors.PATIENT
            elif state == States.DEAD:
                color = UIColors.DEAD
            else:
                raise ValueError("Error: Invalid state!")
            self.window.addstr(y + 1, x * 2 + 1, str(state) + ' ',
                               curses.color_pair(color) | curses.A_BOLD)
        self.window.border()


class StatWindow(BaseSubWindow):
    HEIGHT = 8 - 2  # minus borders
    WIDTH = 20 - 2  # minus borders
    X = 15
    Y = 3
    TITLE_LINE = 1  # plus border
    TITLE_TEXT = "Statistics"
    NORM_LINE = TITLE_LINE + 2
    NORM_TEXT = "Normal:"
    INF_LINE = NORM_LINE + 1
    INF_TEXT = "Infected:"
    ILL_LINE = INF_LINE + 1
    ILL_TEXT = "Patient:"
    DEAD_LINE = ILL_LINE + 1
    DEAD_TEXT = "Dead:"

    def __init__(self, stdscr, field: Field):
        super().__init__(stdscr, StatWindow.HEIGHT + 2,
                         StatWindow.WIDTH + 2, StatWindow.Y, StatWindow.X)
        self.field: Field = field

    @staticmethod
    def render_int(num: int) -> str:
        if num < 10_000:
            return str(num)
        elif num < 1000_000:
            return f"{num // 1000}k"
        else:
            return f"{num // 1000_000}M"

    def render_window(self) -> Statistics:
        return self.field.get_statistics()

    def window_formatting(self):
        stats: Statistics = self.render_window()
        draw_text = lambda pos, text: self.window.addstr(pos, 1, text,
                                                         curses.color_pair(
                                                             UIColors.TEXT) | curses.A_BOLD)
        add_text = lambda pos, left_text, num, color: self.window.addstr(pos,
                                                                         len(
                                                                             left_text) + 1,
                                                                         self.render_int(
                                                                             num).rjust(
                                                                             StatWindow.WIDTH - len(
                                                                                 left_text)),
                                                                         curses.color_pair(
                                                                             color) | curses.A_BOLD)
        draw_text(StatWindow.TITLE_LINE,
                  StatWindow.TITLE_TEXT.center(StatWindow.WIDTH))

        draw_text(StatWindow.NORM_LINE, StatWindow.NORM_TEXT)
        add_text(StatWindow.NORM_LINE, StatWindow.NORM_TEXT, stats.normal,
                 UIColors.NORMAL)

        draw_text(StatWindow.INF_LINE, StatWindow.INF_TEXT)
        add_text(StatWindow.INF_LINE, StatWindow.INF_TEXT, stats.infected,
                 UIColors.INFECTED)

        draw_text(StatWindow.ILL_LINE, StatWindow.ILL_TEXT)
        add_text(StatWindow.ILL_LINE, StatWindow.ILL_TEXT, stats.patient,
                 UIColors.PATIENT)

        draw_text(StatWindow.DEAD_LINE, StatWindow.DEAD_TEXT)
        add_text(StatWindow.DEAD_LINE, StatWindow.DEAD_TEXT, stats.dead,
                 UIColors.DEAD)

        self.window.border()


class GameUI:
    def __init__(self, field: Field, era_num: int, delay_ms: int = 200):
        self.field: Field = field
        self.current_era: int = 0
        self.last_era: int = era_num
        self.delay_ms: int = delay_ms

        self.screen = None
        self.progress_window: ProgressBar
        self.field_window: FieldWindow
        self.statistics_window: StatWindow

    def _update_screen(self):
        self.screen.clear()  # Clear screen
        self.screen.refresh()
        self.field_window.refresh()
        self.progress_window.refresh(self.current_era / self.last_era)
        self.statistics_window.refresh()

        curses.napms(self.delay_ms)

    def _ui_process(self):
        self.screen.nodelay(True)
        self.progress_window = ProgressBar(self.screen)
        self.field_window = FieldWindow(self.screen, self.field)
        self.statistics_window = StatWindow(self.screen, self.field)

        while self.current_era <= self.last_era and self.screen.getch() == -1:
            self._update_screen()
            self.field.change_the_era()
            self.current_era += 1

        self.screen.nodelay(False)
        self.screen.getch()

    def start_ui(self):
        self.screen = curses.initscr()
        # disable cursor
        curses.curs_set(0)
        # Start colors in curses
        curses.start_color()
        curses.init_pair(UIColors.NORMAL, curses.COLOR_GREEN,
                         curses.COLOR_BLACK)
        curses.init_pair(UIColors.INFECTED, curses.COLOR_YELLOW,
                         curses.COLOR_BLACK)
        curses.init_pair(UIColors.PATIENT, curses.COLOR_RED,
                         curses.COLOR_BLACK)
        curses.init_pair(UIColors.DEAD, curses.COLOR_WHITE, curses.COLOR_BLACK)
        try:
            self._ui_process()
        finally:
            curses.endwin()


def main(stdscr):
    i = 0
    limit = 100
    progress = ProgressBar(stdscr)
    stdscr.nodelay(True)
    while stdscr.getch() == -1 and i <= limit:
        stdscr.clear()  # Clear screen

        stdscr.addstr(f'{i} New line. Window size(w = {stdscr.getmaxyx()[1]}, '
                      f'h = {stdscr.getmaxyx()[0]})')

        stdscr.refresh()
        progress.refresh(i / limit)
        curses.delay_output(50)
        i += 1
    stdscr.nodelay(False)
    stdscr.getch()


if __name__ == '__main__':
    curses.wrapper(main)
