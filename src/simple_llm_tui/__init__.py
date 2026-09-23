import asyncio

from .client import async_main


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
