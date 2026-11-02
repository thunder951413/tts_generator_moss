# coding=utf-8
"""HTML rendering for the service login page."""

from __future__ import annotations

import html as html_lib


def _login_html(*, next_path: str, error: str) -> str:
    safe_next = html_lib.escape(next_path, quote=True)
    safe_error = html_lib.escape(error)
    error_block = f'<div class="error">{safe_error}</div>' if safe_error else ""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3-TTS 服务登录</title><style>
body{{margin:0;background:#f3f4f6;font-family:Inter,"Microsoft YaHei",sans-serif;color:#171717;display:grid;place-items:center;min-height:100vh}}
.card{{width:min(420px,calc(100vw - 40px));background:#fff;border:1px solid #ddd;border-radius:12px;padding:28px;box-shadow:0 12px 35px #00000012}}
h1{{font-size:22px;margin:0 0 8px}}p{{color:#666;margin:0 0 22px}}label{{display:block;font-weight:700;margin-bottom:8px}}
input{{width:100%;box-sizing:border-box;padding:11px;border:1px solid #bbb;border-radius:7px;font-size:16px}}
button{{width:100%;margin-top:16px;padding:11px;border:0;border-radius:7px;background:#166534;color:#fff;font-weight:700;font-size:15px;cursor:pointer}}
.error{{color:#b91c1c;background:#fef2f2;padding:9px;border-radius:6px;margin-bottom:14px}}
</style></head><body><form class="card" method="post" action="/login">
<h1>Qwen3-TTS 服务登录</h1><p>输入服务密码后可查看和管理所有生成任务。</p>{error_block}
<input type="hidden" name="next_path" value="{safe_next}"><label for="password">服务密码</label>
<input id="password" name="password" type="password" autocomplete="current-password" required autofocus>
<button type="submit">登录</button></form></body></html>"""
