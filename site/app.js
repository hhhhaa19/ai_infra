(function () {
  var VIEWS = ["overview", "notes", "rmsnorm", "source", "leetcode"];
  var navLinks = document.querySelectorAll("nav.views a");

  function showView(name) {
    VIEWS.forEach(function (v) { document.getElementById(v).hidden = v !== name; });
    navLinks.forEach(function (a) { if (a.dataset.view === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  }
  function showModule(id) {
    document.querySelectorAll("article.mod").forEach(function (m) { m.hidden = m.id !== id; });
    document.querySelectorAll(".mod-list button").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.mod === id)); });
  }
  function route() {
    var id = (location.hash || "#overview").slice(1);
    if (!/^[A-Za-z0-9_.~-]+$/.test(id)) id = "overview";
    var el = document.getElementById(id);
    if (!el) { showView("overview"); return; }
    if (VIEWS.indexOf(id) >= 0) { showView(id); window.scrollTo(0, 0); return; }
    var view = el.closest(".view");
    if (view) showView(view.id);
    var mod = el.closest("article.mod");
    if (mod) showModule(mod.id);
    if (el.tagName === "DETAILS") el.open = true;
    requestAnimationFrame(function () { el.scrollIntoView(); });
  }
  window.addEventListener("hashchange", route);

  // 复习笔记：模块切换
  document.querySelectorAll(".mod-list button").forEach(function (b) {
    b.addEventListener("click", function () { showModule(b.dataset.mod); });
  });

  // 面试追问：筛选 + 已掌握（只存在当前浏览器）
  var KEY = "ai-infra-log-mastered";
  var saved = {};
  try { saved = JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch (e) { saved = {}; }
  var cards = Array.prototype.slice.call(document.querySelectorAll("details.qa"));
  var qf = "all";
  function save() { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {} }
  function paint(c) {
    var d = !!saved[c.id];
    c.classList.toggle("done", d);
    c.querySelector(".mark").textContent = d ? "已掌握" : "";
    c.querySelector("input[type=checkbox]").checked = d;
  }
  function filterQa() {
    var shown = 0, done = 0;
    cards.forEach(function (c) {
      var d = !!saved[c.id]; if (d) done++;
      var ok = qf === "all" || (qf === "todo" ? !d : c.dataset.t === qf);
      c.hidden = !ok; if (ok) shown++;
    });
    document.getElementById("qa-count").textContent = "已掌握 " + done + " / " + cards.length + (qf === "all" ? "" : " · 显示 " + shown);
  }
  cards.forEach(function (c) {
    paint(c);
    c.querySelector("input[type=checkbox]").addEventListener("change", function (ev) {
      if (ev.target.checked) saved[c.id] = 1; else delete saved[c.id];
      save(); paint(c); filterQa();
    });
  });
  document.querySelectorAll(".qf").forEach(function (b) {
    b.addEventListener("click", function () {
      qf = b.dataset.f;
      document.querySelectorAll(".qf").forEach(function (o) { o.setAttribute("aria-pressed", String(o === b)); });
      filterQa();
    });
  });
  filterQa();

  route();
})();
;
// 11-07 新增：复习笔记里点左侧模块，页面跳到该模块的标题处
(function () {
  document.querySelectorAll(".mod-list button").forEach(function (b) {
    b.addEventListener("click", function () {
      var art = document.getElementById(b.dataset.mod);
      if (!art) return;
      var head = document.querySelector("header.top");
      var off = (head ? head.getBoundingClientRect().height : 0) + 16;
      var y = art.getBoundingClientRect().top + window.pageYOffset - off;
      window.scrollTo({ top: Math.max(0, y), behavior: "instant" });
    });
  });
})();
;
// 11-07 新增：把左侧模块按钮里的“11 · ”拆成编号徽标，论文模块用“论”
(function () {
  document.querySelectorAll(".mod-list button").forEach(function (b) {
    var t = b.textContent, m = t.match(/^(\d+)\s*·\s*(.*)$/);
    var num = m ? m[1] : "论", name = m ? m[2] : t;
    b.textContent = "";
    var n = document.createElement("span"); n.className = "mnum"; n.setAttribute("aria-hidden", "true"); n.textContent = num;
    var l = document.createElement("span"); l.textContent = name;
    if (m) l.setAttribute("aria-label", num + " " + name);
    b.appendChild(n); b.appendChild(l);
  });
})();
;
// 11-09 新增：刷题视图。进度存在本页的 db（集合 leetcode，文档 id = 题号），只有页面所有者能勾选。
(function () {
  var CATS = [["哈希", [[1, "两数之和", "E"], [49, "字母异位词分组", "M"], [128, "最长连续序列", "M"]]], ["双指针", [[283, "移动零", "E"], [11, "盛最多水的容器", "M"], [15, "三数之和", "M"], [42, "接雨水", "H"]]], ["滑动窗口", [[3, "无重复字符的最长子串", "M"], [438, "找到字符串中所有字母异位词", "M"]]], ["子串", [[560, "和为 K 的子数组", "M"], [239, "滑动窗口最大值", "H"], [76, "最小覆盖子串", "H"]]], ["普通数组", [[53, "最大子数组和", "M"], [56, "合并区间", "M"], [189, "轮转数组", "M"], [238, "除了自身以外数组的乘积", "M"], [41, "缺失的第一个正数", "H"]]], ["矩阵", [[73, "矩阵置零", "M"], [54, "螺旋矩阵", "M"], [48, "旋转图像", "M"], [240, "搜索二维矩阵 II", "M"]]], ["链表", [[160, "相交链表", "E"], [206, "反转链表", "E"], [234, "回文链表", "E"], [141, "环形链表", "E"], [142, "环形链表 II", "M"], [21, "合并两个有序链表", "E"], [2, "两数相加", "M"], [19, "删除链表的倒数第 N 个结点", "M"], [24, "两两交换链表中的节点", "M"], [25, "K 个一组翻转链表", "H"], [138, "随机链表的复制", "M"], [148, "排序链表", "M"], [23, "合并 K 个升序链表", "H"], [146, "LRU 缓存", "M"]]], ["二叉树", [[94, "二叉树的中序遍历", "E"], [104, "二叉树的最大深度", "E"], [226, "翻转二叉树", "E"], [101, "对称二叉树", "E"], [543, "二叉树的直径", "E"], [102, "二叉树的层序遍历", "M"], [108, "将有序数组转换为二叉搜索树", "E"], [98, "验证二叉搜索树", "M"], [230, "二叉搜索树中第 K 小的元素", "M"], [199, "二叉树的右视图", "M"], [114, "二叉树展开为链表", "M"], [105, "从前序与中序遍历序列构造二叉树", "M"], [437, "路径总和 III", "M"], [236, "二叉树的最近公共祖先", "M"], [124, "二叉树中的最大路径和", "H"]]], ["图论", [[200, "岛屿数量", "M"], [994, "腐烂的橘子", "M"], [207, "课程表", "M"], [208, "实现 Trie (前缀树)", "M"]]], ["回溯", [[46, "全排列", "M"], [78, "子集", "M"], [17, "电话号码的字母组合", "M"], [39, "组合总和", "M"], [22, "括号生成", "M"], [79, "单词搜索", "M"], [131, "分割回文串", "M"], [51, "N 皇后", "H"]]], ["二分查找", [[35, "搜索插入位置", "E"], [74, "搜索二维矩阵", "M"], [34, "在排序数组中查找元素的第一个和最后一个位置", "M"], [33, "搜索旋转排序数组", "M"], [153, "寻找旋转排序数组中的最小值", "M"], [4, "寻找两个正序数组的中位数", "H"]]], ["栈", [[20, "有效的括号", "E"], [155, "最小栈", "M"], [394, "字符串解码", "M"], [739, "每日温度", "M"], [84, "柱状图中最大的矩形", "H"]]], ["堆", [[215, "数组中的第K个最大元素", "M"], [347, "前 K 个高频元素", "M"], [295, "数据流的中位数", "H"]]], ["贪心算法", [[121, "买卖股票的最佳时机", "E"], [55, "跳跃游戏", "M"], [45, "跳跃游戏 II", "M"], [763, "划分字母区间", "M"]]], ["动态规划", [[70, "爬楼梯", "E"], [118, "杨辉三角", "E"], [198, "打家劫舍", "M"], [279, "完全平方数", "M"], [322, "零钱兑换", "M"], [139, "单词拆分", "M"], [300, "最长递增子序列", "M"], [152, "乘积最大子数组", "M"], [416, "分割等和子集", "M"], [32, "最长有效括号", "H"]]], ["多维动态规划", [[62, "不同路径", "M"], [64, "最小路径和", "M"], [5, "最长回文子串", "M"], [1143, "最长公共子序列", "M"], [72, "编辑距离", "M"]]], ["技巧", [[136, "只出现一次的数字", "E"], [169, "多数元素", "E"], [75, "颜色分类", "M"], [31, "下一个排列", "M"], [287, "寻找重复数", "M"]]]];
  var START = "2026-11-10", PER_DAY = 2;
  var done = {};            // 题号 -> 完成日期
  var canWrite = false, db = null;
  function today() { var d = new Date(); return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0"); }
  function daysBetween(a, b) { return Math.round((new Date(b + "T00:00:00") - new Date(a + "T00:00:00")) / 86400000); }
  var DN = { E: "简单", M: "中等", H: "困难" };
  var root = document.getElementById("lc-cats");
  CATS.forEach(function (c) {
    var box = document.createElement("div"); box.className = "lc-cat";
    var h = document.createElement("h3"); h.textContent = c[0]; var cnt = document.createElement("span"); cnt.dataset.cat = c[0]; h.appendChild(cnt); box.appendChild(h);
    c[1].forEach(function (p) {
      var row = document.createElement("label"); row.className = "lc-item"; row.dataset.no = p[0];
      var cb = document.createElement("input"); cb.type = "checkbox"; cb.disabled = true; cb.setAttribute("aria-label", p[0] + ". " + p[1]);
      cb.addEventListener("change", function () { toggle(String(p[0]), cb); });
      var no = document.createElement("span"); no.className = "no"; no.textContent = p[0];
      var a = document.createElement("a"); a.href = "https://leetcode.cn/problemset/?search=" + p[0]; a.target = "_blank"; a.rel = "noopener"; a.textContent = p[1];
      a.addEventListener("click", function (e) { e.stopPropagation(); });
      var d = document.createElement("span"); d.className = "lc-d " + p[2]; d.textContent = DN[p[2]];
      var at = document.createElement("span"); at.className = "at";
      row.append(cb, no, a, d, at); box.appendChild(row);
    });
    root.appendChild(box);
  });
  function render() {
    var n = 0, t = today(), td = 0, wk = 0;
    Object.keys(done).forEach(function (k) { n++; if (done[k] === t) td++; var g = daysBetween(done[k], t); if (g >= 0 && g < 7) wk++; });
    root.querySelectorAll(".lc-item").forEach(function (row) {
      var k = row.dataset.no, cb = row.querySelector("input");
      cb.checked = !!done[k]; cb.disabled = !canWrite; row.classList.toggle("done", !!done[k]);
      row.querySelector(".at").textContent = done[k] ? done[k].slice(5) : "";
    });
    CATS.forEach(function (c) { var m = c[1].filter(function (p) { return done[String(p[0])]; }).length; root.querySelector('[data-cat="' + c[0] + '"]').textContent = m + " / " + c[1].length; });
    var planned = Math.max(0, Math.min(100, (daysBetween(START, t) + 1) * PER_DAY));
    if (t < START) planned = 0;
    document.getElementById("lc-done").textContent = n + " / 100";
    document.getElementById("lc-plan").textContent = planned;
    document.getElementById("lc-today").textContent = td;
    document.getElementById("lc-week").textContent = wk;
    document.getElementById("lc-bar-fill").style.width = n + "%";
    document.getElementById("lc-bar-plan").style.left = "calc(" + planned + "% - 1px)";
  }
  function setStatus(msg) { document.getElementById("lc-status").textContent = msg; }
  function toggle(k, cb) {
    if (!db || !canWrite) return;
    var ref = db.collection("leetcode").doc(k);
    cb.disabled = true;
    var p = cb.checked ? ref.set({ done: true, at: today() }) : ref.delete();
    p.then(function () { cb.disabled = false; }, function (e) {
      cb.checked = !cb.checked; cb.disabled = false;
      setStatus(e && e.code === "invalid_argument" ? "没有保存：只有页面所有者能勾选。" : "没有保存，请稍后再试。");
      if (e && e.code === "invalid_argument") { canWrite = false; render(); }
    });
  }
  render();
  if (!window.claude || !window.claude.use) { setStatus("当前环境读不到进度存储，只显示题单。"); return; }
  Promise.all([window.claude.use("db"), window.claude.use("user")]).then(function (r) {
    db = r[0]; var user = r[1];
    if (!db) { setStatus("当前环境读不到进度存储（需要在 claude.ai 登录后打开），只显示题单。"); return; }
    canWrite = user ? !!user.isOwner() : false;
    var timer = setTimeout(function () { setStatus("进度暂时读不到，显示的可能不是最新的。"); }, 10000);
    db.collection("leetcode").onSnapshot(function (snap) {
      var next = {};
      snap.docs.forEach(function (d) { var v = d.data(); if (v && v.done) next[d.id] = v.at || ""; });
      done = next; render();
      if (!snap.metadata.fromCache) { clearTimeout(timer); setStatus(canWrite ? "勾选即保存，记下当天日期；题名链接到力扣中国站。" : "只读：只有页面所有者能勾选。"); }
    }, function () { clearTimeout(timer); setStatus("进度读取失败，只显示题单。"); });
  });
})();
