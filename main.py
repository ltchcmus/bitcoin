def main() -> None:
    import sys

    from trading_pipeline.cli import main as cli_main
    from trading_pipeline.gui_app import run_gui

    if len(sys.argv) > 1:
        cli_main()
    else:
        run_gui()


if __name__ == "__main__":
    main()