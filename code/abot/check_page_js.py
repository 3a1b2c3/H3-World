#!/usr/bin/env python3
"""在假 DOM 下真跑一遍页面的 <script>，确认渲染路径没断。

写这个是因为踩过一次：`padsHtml` 里引用的 `const KIDX` 声明在渲染代码之后，
函数声明会提升而 `const` 不会（暂时性死区），调用时抛 ReferenceError，
整个 script 挂掉、页面一片空白。而"占位符已替换 / 标签配平 / id 都存在"
这类静态检查全是绿的 —— 静态检查验不出运行时的东西。

用法:
    python3 code/abot/check_page_js.py docs/action_prompt_viz.html
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

STUB = r"""
// 最小 DOM 桩：脚本能跑完且往目标容器写进了内容，就说明渲染路径通。
const store = {};
const mkEl = (id) => ({
  _html: "", id,
  set innerHTML(v) { this._html = v; store[id] = v; },
  get innerHTML() { return this._html; },
  set textContent(v) { store[id] = v; },
  querySelectorAll: () => [], addEventListener: () => {},
  classList: { toggle() {}, add() {}, remove() {} },
});
const els = {};
global.document = {
  getElementById: (id) => (els[id] = els[id] || mkEl(id)),
  querySelectorAll: () => [], querySelector: () => null,
  addEventListener: () => {},
};
require(process.argv[2]);
console.log(JSON.stringify(Object.fromEntries(
  Object.entries(store).map(([k, v]) => [k, String(v).length]))));
"""


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    page = Path(sys.argv[1])
    html = page.read_text(encoding="utf-8")
    if "<script>" not in html:
        print("页面没有 <script>，跳过")
        return
    js = html[html.index("<script>") + len("<script>"): html.rindex("</script>")]

    with tempfile.TemporaryDirectory() as td:
        jsf = Path(td) / "page.js"
        runner = Path(td) / "run.js"
        jsf.write_text(js, encoding="utf-8")
        runner.write_text(STUB, encoding="utf-8")
        r = subprocess.run(["node", str(runner), str(jsf)],
                           capture_output=True, text=True)
    if r.returncode != 0:
        print("✗ 脚本抛异常：\n" + (r.stderr.strip() or r.stdout.strip()))
        sys.exit(1)

    written = json.loads(r.stdout.strip().splitlines()[-1])
    empty = [k for k, n in written.items() if n == 0]
    print(f"✓ 脚本执行完毕，写入 {len(written)} 个容器")
    for k, n in sorted(written.items(), key=lambda x: -x[1]):
        print(f"    {k:<10} {n:>10,} 字符")
    if empty:
        print(f"✗ 这些容器没写进内容: {empty}")
        sys.exit(1)


if __name__ == "__main__":
    main()
