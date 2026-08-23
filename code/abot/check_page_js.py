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
//
// 桩要撑到什么程度，是被页面用到的 API 决定的 —— 只支持 innerHTML 的版本
// 跑不了 createElement/appendChild 建树的页面，检查器会假报"脚本抛异常"。
// 所以这里补齐了建树那一套，并且**选择器要真的去匹配 innerHTML**：
// querySelector 匹配不上就返回 null，让写错的选择器当场暴露 —— 这正是
// 静态检查抓不到、而线上表现为"某块区域空白"的那类错误。
const store = {};

function selRe(sel) {
  // 只取最后一个简单选择器：'.said .t' 这类后代选择器在真浏览器里没问题，
  // 桩里不实现完整的层级匹配，否则要建真 DOM 树。代价是祖先写错了这里查不出来，
  // 但"整条渲染路径能不能跑完"这个目的达到了 —— 别让判据自己成为误报源。
  const s = sel.trim().split(/\s+/).pop();
  if (s.startsWith('.')) return new RegExp('class="[^"]*\\b' + s.slice(1) + '\\b[^"]*"', 'g');
  if (s.startsWith('#')) return new RegExp('id="' + s.slice(1) + '"', 'g');
  return new RegExp('<' + s + '[\\s>]', 'g');
}

function mkNode(id) {
  const node = {
    id, _html: "", children: [], style: {}, dataset: {},
    set innerHTML(v) { this._html = String(v); if (id) store[id] = this._html; },
    get innerHTML() { return this._html; },
    set textContent(v) { this._text = String(v); if (id) store[id] = this._text; },
    get textContent() { return this._text || ""; },
    appendChild(c) {
      this.children.push(c);
      if (id) store[id] = (store[id] || "") + (c._html || "") + (c._text || "");
      return c;
    },
    querySelectorAll(sel) {
      const n = (this._html.match(selRe(sel)) || []).length;
      return Array.from({ length: n }, () => mkNode(null));
    },
    querySelector(sel) {
      return selRe(sel).test(this._html) ? mkNode(null) : null;
    },
    addEventListener() {}, removeEventListener() {}, remove() {}, focus() {},
    getAttribute: () => null, setAttribute() {},
    classList: {
      _s: new Set(),
      add(...c) { c.forEach(x => this._s.add(x)); },
      remove(...c) { c.forEach(x => this._s.delete(x)); },
      toggle(c) { this._s.has(c) ? this._s.delete(c) : this._s.add(c); },
      contains(c) { return this._s.has(c); },
    },
  };
  return node;
}

const els = {};
global.document = {
  getElementById: (id) => (els[id] = els[id] || mkNode(id)),
  createElement: () => mkNode(null),
  createTextNode: () => mkNode(null),
  querySelectorAll: () => [], querySelector: () => null,
  addEventListener: () => {},
  body: mkNode(null),
};
global.window = { addEventListener: () => {}, matchMedia: () => ({ matches: false, addEventListener() {} }) };
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
