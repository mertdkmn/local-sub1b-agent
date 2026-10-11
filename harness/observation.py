"""Build the compact observation the agent sees at each step.

RESEARCH_PLAN §8: the agent never sees raw HTML — it sees a compact list of
*interactive* elements only. Two steps:

1. ``observe`` injects a stable ``data-ref="N"`` attribute onto every actionable
   element in DOM order, then reads back just the fields an agent needs to decide
   on an action (tag, role, text, type, placeholder, value, href, disabled).
2. ``render`` flattens those into the text form that the real model (v2) will
   consume as its prompt.

The ``data-ref`` trick is what makes CLICK(ref)/TYPE(ref, text) deterministic:
the ref is a live index into the current page, not a guessable CSS selector.
"""
from __future__ import annotations

from dataclasses import dataclass

from playwright.sync_api import Page

# Which elements count as "actionable". Paragraphs, headings, images etc. are
# deliberately excluded — the agent can't act on them, so showing them is noise.
_TAG_SELECTOR = (
    "a, button, input, select, textarea, "
    "[role='button'], [role='link'], [role='textbox'], "
    "[role='checkbox'], [role='combobox'], [role='option'], [tabindex]"
)

_JS_TAG = f"""
() => {{
  const els = document.querySelectorAll({_TAG_SELECTOR!r});
  els.forEach((el, i) => el.setAttribute('data-ref', String(i)));
  return els.length;
}}
"""

_JS_READ = f"""
() => {{
  // Best-effort accessible label for a form control: aria-label, then
  // aria-labelledby, then <label for="id">, then a wrapping <label>. This is
  // what lets a field like <input id="name"> (no placeholder) show up as
  // "Name" instead of a bare type string.
  const labelFor = (el) => {{
    const aria = el.getAttribute('aria-label');
    if (aria) return aria.trim();
    const by = el.getAttribute('aria-labelledby');
    if (by) {{
      const ref = document.getElementById(by);
      if (ref) return (ref.innerText || ref.textContent || '').trim();
    }}
    if (el.id) {{
      const lab = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (lab) return (lab.innerText || lab.textContent || '').trim();
    }}
    const wrap = el.closest('label');
    if (wrap) return (wrap.innerText || wrap.textContent || '').trim();
    return '';
  }};
  return Array.from(document.querySelectorAll({_TAG_SELECTOR!r})).map(el => ({{
    ref: el.getAttribute('data-ref'),
    tag: el.tagName.toLowerCase(),
    role: el.getAttribute('role') || '',
    text: (el.innerText || el.textContent || '').trim().slice(0, 80),
    label: labelFor(el).slice(0, 80),
    type: el.getAttribute('type') || '',
    placeholder: el.getAttribute('placeholder') || '',
    value: el.value || '',
    href: el.getAttribute('href') || '',
    disabled: el.hasAttribute('disabled') || el.getAttribute('aria-disabled') === 'true',
  }}));
}}
"""


@dataclass(frozen=True)
class Element:
    ref: int
    tag: str
    role: str
    text: str
    label: str
    type: str
    placeholder: str
    value: str
    href: str
    disabled: bool


def observe(page: Page) -> list[Element]:
    """Tag interactive elements and return them in DOM order."""
    page.evaluate(_JS_TAG)
    rows = page.evaluate(_JS_READ)
    return [
        Element(
            ref=int(r["ref"]),
            tag=r["tag"],
            role=r["role"],
            text=r["text"],
            label=r["label"],
            type=r["type"],
            placeholder=r["placeholder"],
            value=r["value"],
            href=r["href"],
            disabled=r["disabled"],
        )
        for r in rows
    ]


def render(elements: list[Element]) -> str:
    """Flatten the element list into the agent's text observation.

    The v1 fake agent ignores this string, but building it now pins the exact
    format the real model will be prompted with, so trajectories collected later
    (P1) are consistent.
    """
    lines: list[str] = []
    for el in elements:
        # Priority: the element's own text, then its accessible label, then
        # placeholder/value/href, then type, then tag as a last resort.
        label = el.text or el.label or el.placeholder or el.value or el.href or el.type or el.tag
        # A filled form field shows its current value so the agent can see its
        # own progress: "Name=John Doe" vs the still-empty "Name".
        if el.value and el.tag in ("input", "textarea") and (el.label or el.placeholder):
            label = f"{el.label or el.placeholder}={el.value}"
        lines.append(f"[{el.ref}] <{el.tag}> {label}")
    return "\n".join(lines)
