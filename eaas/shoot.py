import asyncio
from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:5055"
OUT = "/home/claude/eaas/screenshots"

PAGES = [
    ("home", "/"),
    ("register_form", "/register"),
    ("login_scan", "/login"),
    ("register_success", "/register/success/1"),
    ("result_granted", "/result/1"),
    ("result_flagged", "/result/4"),
    ("result_denied", "/result/5"),
    ("admin_logs", "/admin/logs"),
    ("admin_users", "/admin/users"),
]


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=[
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
        ])
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        await context.grant_permissions(["camera"])
        page = await context.new_page()
        for name, path in PAGES:
            await page.goto(f"{BASE}{path}", wait_until="networkidle")
            if name == "register_form":
                await page.fill("#full_name", "Oyewunmi Demilade Peter")
                await page.fill("#matric_no", "2021002681")
                await page.fill("#department", "Computer Science")
                await page.fill("#email", "demilade@lautech.edu.ng")
            await page.wait_for_timeout(700)
            full = name in ("home", "admin_logs", "admin_users")
            await page.screenshot(path=f"{OUT}/{name}.png", full_page=full)
            print("captured", name)
        await browser.close()

asyncio.run(main())
