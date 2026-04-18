"""Small application entry point for the Krogh GUI project."""

from krogh_GUI import KroghGUI


def main() -> None:
    app = KroghGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
