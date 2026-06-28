// expr.js — parse a complex analytic function f(z) into a GLSL snippet.
//
// OWNED BY AGENT B. Used by field.js for the `complex` field. The Python
// reference (lic_main.py / lic_fields.complex_func) evaluates a numpy lambda
// like `z**2`, `1/z`, `np.sin(z*4)`, `(z-0.5-0.5j)**3` over z = x + 1j*y and
// returns (Re(w), Im(w)) as the field, then normalises to unit length.
//
// We don't have numpy in the shader, so we parse the expression here (recursive
// descent + complex arithmetic codegen) and emit a GLSL expression operating on
// `vec2` (x=real, y=imag). The caller injects it as `cexpr(z)` in the fragment
// shader, where the complex helper functions (cmul, cdiv, cpow, csin, ...) are
// provided by fields-eval.frag.
//
// Supported grammar:
//   expr    := term (('+'|'-') term)*
//   term    := unary (('*'|'/') unary)*
//   unary   := ('+'|'-') unary | power
//   power   := atom ('**' unary)?            (right-assoc, like Python)
//   atom    := number | imag | 'z' | func '(' expr ')' | '(' expr ')'
//   func    := sin | cos | exp | conj
//   number  := real literal (may be followed by 'j' to make it imaginary)
//   imag    := number 'j'   (e.g. 0.5j, 2j, 1j)
//
// `np.` prefixes are tolerated (np.sin -> sin). Unsupported tokens raise; the
// caller (field.js) falls back to a safe default (z**2).

// ---- tokenizer -------------------------------------------------------------

function tokenize(src) {
  const tokens = [];
  let i = 0;
  const s = src.replace(/np\./g, ""); // tolerate numpy prefixes
  const n = s.length;
  const isDigit = (c) => c >= "0" && c <= "9";
  const isAlpha = (c) => (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || c === "_";
  while (i < n) {
    const c = s[i];
    if (c === " " || c === "\t" || c === "\n" || c === "\r") { i++; continue; }
    if (c === "*" && s[i + 1] === "*") { tokens.push({ t: "pow" }); i += 2; continue; }
    if ("+-*/(),".includes(c)) { tokens.push({ t: c }); i++; continue; }
    if (isDigit(c) || (c === "." && isDigit(s[i + 1]))) {
      let j = i;
      while (j < n && (isDigit(s[j]) || s[j] === ".")) j++;
      // scientific notation: 1e-3
      if (j < n && (s[j] === "e" || s[j] === "E")) {
        j++;
        if (j < n && (s[j] === "+" || s[j] === "-")) j++;
        while (j < n && isDigit(s[j])) j++;
      }
      const num = parseFloat(s.slice(i, j));
      i = j;
      if (i < n && (s[i] === "j" || s[i] === "J")) {
        i++;
        tokens.push({ t: "num", re: 0, im: num });
      } else {
        tokens.push({ t: "num", re: num, im: 0 });
      }
      continue;
    }
    if (isAlpha(c)) {
      let j = i;
      while (j < n && (isAlpha(s[j]) || isDigit(s[j]))) j++;
      const word = s.slice(i, j);
      i = j;
      // bare 'j' as imaginary unit
      if (word === "j" || word === "J") { tokens.push({ t: "num", re: 0, im: 1 }); continue; }
      if (word === "z") { tokens.push({ t: "z" }); continue; }
      if (["sin", "cos", "exp", "conj"].includes(word)) { tokens.push({ t: "func", name: word }); continue; }
      throw new Error("Unknown identifier in expr: " + word);
    }
    throw new Error("Unexpected character in expr: " + c);
  }
  return tokens;
}

// ---- parser (recursive descent) -------------------------------------------
// AST nodes carry a `glsl` string already; we codegen bottom-up.

function parseExpr(src) {
  const tokens = tokenize(src);
  let pos = 0;
  const peek = () => tokens[pos];
  const next = () => tokens[pos++];
  const expect = (t) => {
    const tok = next();
    if (!tok || tok.t !== t) throw new Error("Expected '" + t + "' in expr");
    return tok;
  };

  function fmtFloat(x) {
    let s = String(x);
    if (!/[.eE]/.test(s)) s += ".0";
    return s;
  }
  // emit a vec2 literal for a complex constant
  function constVec(re, im) {
    return `vec2(${fmtFloat(re)}, ${fmtFloat(im)})`;
  }

  function parseExpression() {
    let node = parseTerm();
    while (peek() && (peek().t === "+" || peek().t === "-")) {
      const op = next().t;
      const rhs = parseTerm();
      node = op === "+" ? `(${node} + ${rhs})` : `(${node} - ${rhs})`;
    }
    return node;
  }

  function parseTerm() {
    let node = parseUnary();
    while (peek() && (peek().t === "*" || peek().t === "/")) {
      const op = next().t;
      const rhs = parseUnary();
      node = op === "*" ? `cmul(${node}, ${rhs})` : `cdiv(${node}, ${rhs})`;
    }
    return node;
  }

  function parseUnary() {
    if (peek() && (peek().t === "+" || peek().t === "-")) {
      const op = next().t;
      const operand = parseUnary();
      return op === "-" ? `(-${operand})` : operand;
    }
    return parsePower();
  }

  function parsePower() {
    const base = parseAtom();
    if (peek() && peek().t === "pow") {
      next();
      const exp = parseUnary(); // right-assoc
      return `cpow(${base}, ${exp})`;
    }
    return base;
  }

  function parseAtom() {
    const tok = peek();
    if (!tok) throw new Error("Unexpected end of expr");
    if (tok.t === "num") { next(); return constVec(tok.re, tok.im); }
    if (tok.t === "z") { next(); return "z"; }
    if (tok.t === "(") { next(); const e = parseExpression(); expect(")"); return e; }
    if (tok.t === "func") {
      next();
      expect("(");
      const arg = parseExpression();
      expect(")");
      const fn = { sin: "csin", cos: "ccos", exp: "cexp", conj: "cconj" }[tok.name];
      return `${fn}(${arg})`;
    }
    throw new Error("Unexpected token in expr: " + tok.t);
  }

  const out = parseExpression();
  if (pos !== tokens.length) throw new Error("Trailing tokens in expr");
  return out;
}

// Public: return a GLSL expression string in terms of `z` (a vec2). On any
// parse error, fall back to the supplied default (default "z**2").
export function exprToGlsl(exprSrc, fallback = "z**2") {
  try {
    return { glsl: parseExpr(String(exprSrc)), ok: true, src: exprSrc };
  } catch (e) {
    try {
      return { glsl: parseExpr(fallback), ok: false, error: String(e && e.message), src: fallback };
    } catch (_) {
      // ultimate fallback: identity
      return { glsl: "z", ok: false, error: String(e && e.message), src: "z" };
    }
  }
}
