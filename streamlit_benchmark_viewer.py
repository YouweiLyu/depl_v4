from __future__ import annotations

import io
import json
import math
import re
import zipfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


WORKSPACE_ROOT = Path(__file__).resolve().parent
RESULTS_DIR_NAME = "results_all_final" #"results_final"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PREFERRED_MODEL_ORDER = [
    "nano-banana2",
    "nano-banana-pro",
    "gpt-image-1.5-high",
    "seedream-5-lite",
]


def list_result_directories() -> list[Path]:
    target = WORKSPACE_ROOT / RESULTS_DIR_NAME
    if target.is_dir():
        return [target]
    return []


def read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def load_json_if_exists(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_first_triple_quoted_block(text: str) -> str:
    match = re.search(r'(["\']{3})([\s\S]*?)\1', text)
    return (match.group(2) or "").strip() if match else ""


def extract_prompt_from_function(script_path: Path, function_name: str) -> str:
    source = read_text_if_exists(script_path)
    if not source:
        return ""
    anchor = f"def {function_name}"
    start = source.find(anchor)
    if start < 0:
        return ""
    snippet = source[start : start + 8000]
    prompt = extract_first_triple_quoted_block(snippet)
    if "{{MANDATORY_FOCUS_CLAUSE}}" in prompt:
        prompt = prompt.replace(
            "{{MANDATORY_FOCUS_CLAUSE}}",
            "[Dynamic mandatory focus clause varies by selected aspect and is omitted here.]",
        )
    return prompt


def extract_prompt_from_assignment(script_path: Path, variable_name: str) -> str:
    source = read_text_if_exists(script_path)
    if not source:
        return ""
    pattern = rf"{re.escape(variable_name)}\s*=\s*([\"']{{3}})([\s\S]*?)\1"
    match = re.search(pattern, source)
    return (match.group(2) or "").strip() if match else ""


def extract_prompt_from_line(script_path: Path, variable_name: str) -> str:
    source = read_text_if_exists(script_path)
    if not source:
        return ""
    pattern = rf'{re.escape(variable_name)}\s*=\s*f?["\']([^"\']+)["\']'
    match = re.search(pattern, source)
    return (match.group(1) or "").strip() if match else ""


def infer_meta_prompt(result_dir: Path) -> tuple[str, str, str]:
    name = result_dir.name
    if name == RESULTS_DIR_NAME:
        script = WORKSPACE_ROOT / "enhance_gemini_analysis_new.py"
        return (
            "Meta Prompt",
            extract_prompt_from_function(script, "build_meta_prompt_with_thinking"),
            script.name,
        )
    if name.startswith("results_test"):
        script = WORKSPACE_ROOT / "enhance_gemini_analysis_new.py"
        return (
            "Meta Prompt",
            extract_prompt_from_function(script, "build_meta_analysis_prompt"),
            script.name,
        )
    if name.startswith("results_part1_unified_prompt") or name.startswith("results_part2_unified_prompt"):
        script = WORKSPACE_ROOT / "enhance_unified_prompt.py"
        return (
            "Shared Prompt Template",
            extract_prompt_from_assignment(script, "SHARED_AESTHETIC_PROMPT"),
            script.name,
        )
    if name.startswith("results_part2_composition_only"):
        script = WORKSPACE_ROOT / "enhance_gemini_composition_from_metadata.py"
        prompt = extract_prompt_from_line(script, "final_prompt")
        return ("Composition Prompt Template", prompt, script.name)
    if name.startswith("results_part2"):
        script = WORKSPACE_ROOT / "enhance_gemini_analysis_part2.py"
        return (
            "Meta Prompt",
            extract_prompt_from_function(script, "build_meta_analysis_prompt"),
            script.name,
        )
    script = WORKSPACE_ROOT / "enhance_gemini_analysis.py"
    return (
        "Meta Prompt",
        extract_prompt_from_function(script, "build_meta_analysis_prompt"),
        script.name,
    )


def extract_record_prompt(payload: dict) -> str:
    prompt = str(payload.get("prompt") or "").strip()
    if prompt:
        return prompt

    analysis_raw_text = payload.get("analysis_raw_text")
    if isinstance(analysis_raw_text, str):
        return analysis_raw_text.strip()

    return ""


def extract_record_prompt_zh(payload: dict) -> str:
    prompt_zh = str(payload.get("prompt_zh") or "").strip()
    if prompt_zh:
        return prompt_zh
    return ""


def extract_record_meta_prompt(payload: dict) -> str:
    meta_prompt = str(payload.get("meta_prompt") or "").strip()
    if meta_prompt:
        return meta_prompt
    return ""


def normalize_group_key(parent: Path, stem: str) -> str:
    parent_text = parent.as_posix().strip(".")
    return f"{parent_text}/{stem}".strip("/") if parent_text else stem


def split_result_stem(stem: str) -> tuple[str, str] | None:
    if "_" not in stem:
        return None
    image_stem, model_alias = stem.rsplit("_", 1)
    if not image_stem or not model_alias:
        return None
    return image_stem, model_alias


def infer_source_roots(result_dir: Path) -> list[Path]:
    name = result_dir.name
    ordered: list[Path] = []
    if name == RESULTS_DIR_NAME:
        ordered.append(WORKSPACE_ROOT / "src_final")
        ordered.append(WORKSPACE_ROOT / "src_all_0312")
    if name.startswith("results_test"):
        ordered.append(WORKSPACE_ROOT / "src_final")
    if name.startswith("results_part1"):
        ordered.append(WORKSPACE_ROOT / "src_part1")
    if name.startswith("results_part2"):
        ordered.append(WORKSPACE_ROOT / "src_part2")
    ordered.extend(
        [
            WORKSPACE_ROOT / "src_final",
            WORKSPACE_ROOT / "src_all_0312",
            WORKSPACE_ROOT / "src_part1",
            WORKSPACE_ROOT / "src_part2",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for item in ordered:
        key = str(item)
        if key not in seen and item.exists():
            unique.append(item)
            seen.add(key)
    return unique


def resolve_source_image(source_image_value: str, parent: str, stem: str, result_dir: Path) -> str:
    if source_image_value:
        source_path = Path(source_image_value)
        if not source_path.is_absolute():
            source_path = WORKSPACE_ROOT / source_path
        if source_path.exists():
            return str(source_path)

    relative_parent = Path(parent) if parent else Path()
    for root in infer_source_roots(result_dir):
        for ext in IMAGE_EXTS:
            candidate = root / relative_parent / f"{stem}{ext}"
            if candidate.exists():
                return str(candidate)
    return ""


@st.cache_data(show_spinner=False)
def scan_result_directory(result_dir_str: str) -> dict:
    result_dir = Path(result_dir_str)
    records: dict[str, dict] = {}

    # ── metadata discovery ──────────────────────────────────────────────────
    # Handles both layouts:
    #   Layout A: result_dir/metadata/category/stem_metadata.json
    #   Layout B: result_dir/category/metadata/stem_metadata.json
    for metadata_path in sorted(result_dir.rglob("*_metadata.json")):
        rel = metadata_path.relative_to(result_dir)
        parts = rel.parts
        try:
            meta_idx = list(parts).index("metadata")
        except ValueError:
            continue
        prefix_parts = parts[:meta_idx]
        suffix_parts = parts[meta_idx + 1 : -1]
        stem = metadata_path.stem[: -len("_metadata")]
        parent_parts = prefix_parts + suffix_parts
        parent = Path(*parent_parts) if parent_parts else Path(".")
        group_key = normalize_group_key(parent, stem)
        category = parent_parts[0] if parent_parts else "root"
        parent_str = parent.as_posix() if parent.as_posix() != "." else ""

        payload = load_json_if_exists(metadata_path)
        record = records.setdefault(
            group_key,
            {
                "group_key": group_key,
                "image_stem": stem,
                "category": category,
                "parent": parent_str,
                "prompt": "",
                "prompt_zh": "",
                "source_image": "",
                "metadata_path": "",
                "prompt_path": "",
                "analysis_model": "",
                "analysis_time_utc": "",
                "meta_prompt": "",
                "edited_attributes": [],
                "structured_prompt": {},
                "model_images": {},
            },
        )
        record["metadata_path"] = str(metadata_path)
        record["prompt"] = extract_record_prompt(payload)
        record["prompt_zh"] = extract_record_prompt_zh(payload)
        record["source_image"] = resolve_source_image(
            str(payload.get("source_image") or ""),
            parent_str,
            stem,
            result_dir,
        )
        record["analysis_model"] = str(payload.get("analysis_model") or payload.get("prompt_mode") or "")
        record["analysis_time_utc"] = str(payload.get("analysis_time_utc") or "")
        record["meta_prompt"] = extract_record_meta_prompt(payload)
        attrs = payload.get("edited_attributes")
        record["edited_attributes"] = attrs if isinstance(attrs, list) else []
        structured = payload.get("structured_prompt")
        record["structured_prompt"] = structured if isinstance(structured, dict) else {}

    # ── used_prompts discovery ───────────────────────────────────────────────
    for prompt_path in sorted(result_dir.rglob("*_prompt.txt")):
        rel = prompt_path.relative_to(result_dir)
        parts = rel.parts
        try:
            pu_idx = list(parts).index("used_prompts")
        except ValueError:
            continue
        prefix_parts = parts[:pu_idx]
        suffix_parts = parts[pu_idx + 1 : -1]
        stem = prompt_path.stem[: -len("_prompt")]
        parent_parts = prefix_parts + suffix_parts
        parent = Path(*parent_parts) if parent_parts else Path(".")
        group_key = normalize_group_key(parent, stem)
        category = parent_parts[0] if parent_parts else "root"
        parent_str = parent.as_posix() if parent.as_posix() != "." else ""

        record = records.setdefault(
            group_key,
            {
                "group_key": group_key,
                "image_stem": stem,
                "category": category,
                "parent": parent_str,
                "prompt": "",
                "prompt_zh": "",
                "source_image": "",
                "metadata_path": "",
                "prompt_path": "",
                "analysis_model": "",
                "analysis_time_utc": "",
                "meta_prompt": "",
                "edited_attributes": [],
                "structured_prompt": {},
                "model_images": {},
            },
        )
        record["prompt_path"] = str(prompt_path)
        if not record["prompt"]:
            record["prompt"] = read_text_if_exists(prompt_path).strip()
        if not record["source_image"]:
            record["source_image"] = resolve_source_image("", parent_str, stem, result_dir)

    # ── image scan ──────────────────────────────────────────────────────────
    discovered_aliases: set[str] = set()
    for image_path in sorted(result_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = image_path.relative_to(result_dir)
        # Skip images that live inside a metadata/ or used_prompts/ directory
        if "metadata" in rel.parts[:-1] or "used_prompts" in rel.parts[:-1]:
            continue

        split = split_result_stem(image_path.stem)
        if not split:
            continue
        stem, model_alias = split
        parent = rel.parent
        parent_str = parent.as_posix() if parent.as_posix() != "." else ""
        group_key = normalize_group_key(parent, stem)
        discovered_aliases.add(model_alias)

        record = records.setdefault(
            group_key,
            {
                "group_key": group_key,
                "image_stem": stem,
                "category": parent.parts[0] if parent.parts else "root",
                "parent": parent_str,
                "prompt": "",
                "prompt_zh": "",
                "source_image": resolve_source_image("", parent_str, stem, result_dir),
                "metadata_path": "",
                "prompt_path": "",
                "analysis_model": "",
                "analysis_time_utc": "",
                "meta_prompt": "",
                "edited_attributes": [],
                "structured_prompt": {},
                "model_images": {},
            },
        )
        record["model_images"][model_alias] = str(image_path)

    # Always expose preferred benchmark models so missing outputs are visible as "缺失".
    ordered_models = list(PREFERRED_MODEL_ORDER)
    ordered_models.extend(sorted(discovered_aliases - set(ordered_models)))

    rows = []
    for record in records.values():
        if not record["source_image"]:
            record["source_image"] = resolve_source_image(
                "",
                record["parent"],
                record["image_stem"],
                result_dir,
            )
        record["available_model_count"] = len(record["model_images"])
        record["complete"] = bool(ordered_models) and all(alias in record["model_images"] for alias in ordered_models)
        record["missing_models"] = [alias for alias in ordered_models if alias not in record["model_images"]]
        rows.append(record)

    rows.sort(key=lambda item: item["group_key"].lower())
    return {"rows": rows, "model_aliases": ordered_models}


@st.cache_data(show_spinner=False)
def get_image_size(image_path: str) -> tuple[int, int] | None:
    if not image_path:
        return None
    try:
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return None


def resolve_page_meta_prompt(rows: list[dict], result_dir: Path) -> tuple[str, str, str]:
    for row in rows:
        meta_prompt = str(row.get("meta_prompt") or "").strip()
        if meta_prompt:
            return "Meta Prompt", meta_prompt, str(row.get("metadata_path") or "metadata")
    return infer_meta_prompt(result_dir)


@st.cache_data(show_spinner=False)
def build_group_zip(file_entries: tuple[tuple[str, str], ...]) -> bytes:
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for archive_name, source_path in file_entries:
            path = Path(source_path)
            if path.exists() and path.is_file():
                zip_file.writestr(archive_name, path.read_bytes())
    return memory_file.getvalue()


def build_download_payload(record: dict, model_aliases: list[str]) -> tuple[bytes, str]:
    entries: list[tuple[str, str]] = []
    source_image = record.get("source_image") or ""
    if source_image:
        source_path = Path(source_image)
        entries.append((f"original{source_path.suffix.lower()}", source_image))

    for alias in model_aliases:
        model_path = record["model_images"].get(alias)
        if model_path:
            suffix = Path(model_path).suffix.lower() or ".png"
            entries.append((f"{alias}{suffix}", model_path))

    metadata_path = record.get("metadata_path") or ""
    if metadata_path:
        entries.append(("metadata.json", metadata_path))

    prompt_text = (record.get("prompt") or "").strip()
    if prompt_text:
        entries.append(("prompt.txt", _write_virtual_file(prompt_text)))

    zip_bytes = build_group_zip(tuple(entries))
    file_name = record["group_key"].replace("/", "__") + ".zip"
    return zip_bytes, file_name


@st.cache_data(show_spinner=False)
def _write_virtual_file(content: str) -> str:
    virtual_dir = WORKSPACE_ROOT / ".streamlit_cache"
    virtual_dir.mkdir(exist_ok=True)
    file_path = virtual_dir / f"{abs(hash(content))}.txt"
    if not file_path.exists():
        file_path.write_text(content, encoding="utf-8")
    return str(file_path)


def compute_stats(rows: list[dict], model_aliases: list[str]) -> dict:
    stats = {
        "total": len(rows),
        "complete": sum(1 for row in rows if row["complete"]),
        "with_prompt": sum(1 for row in rows if row["prompt"]),
        "with_source": sum(1 for row in rows if row["source_image"]),
        "per_model": {alias: sum(1 for row in rows if alias in row["model_images"]) for alias in model_aliases},
    }
    return stats


def format_model_label(model_alias: str) -> str:
    label_map = {
        "nano-banana2": "Nano Banana 2",
        "nano-banana-pro": "Nano Banana Pro",
        "gpt-image-1.5-high": "GPT Image 1.5 High",
        "seedream-5-lite": "Seedream 5 Lite",
    }
    return label_map.get(model_alias, model_alias)


def render_meta_prompt(rows: list[dict], result_dir: Path):
    title, prompt, source = resolve_page_meta_prompt(rows, result_dir)
    st.subheader(title)
    cols = st.columns([4, 1])
    cols[0].caption(f"推断来源: {source}")
    cols[1].caption(f"结果目录: {result_dir.name}")
    with st.expander("展开查看 Meta Prompt", expanded=False):
        st.code(prompt or "未能从 metadata 中读取 meta prompt。", language="text", wrap_lines=True)


def render_stats(stats: dict, model_aliases: list[str]):
    base_metrics = st.columns(4)
    base_metrics[0].metric("样本组数", stats["total"])
    base_metrics[1].metric("完整组数", stats["complete"])
    base_metrics[2].metric("有 Prompt", stats["with_prompt"])
    base_metrics[3].metric("有原图", stats["with_source"])

    if model_aliases:
        model_metrics = st.columns(len(model_aliases))
        for index, alias in enumerate(model_aliases):
            model_metrics[index].metric(format_model_label(alias), stats["per_model"][alias])


def inject_resizable_textarea_css():
    st.markdown(
        """
        <style>
        /* Allow users to drag textarea height in the prompt panel. */
        [data-testid="stTextArea"] textarea {
            resize: vertical !important;
            min-height: 120px;
            max-height: 720px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_prompt_hover_tooltip_js():
        components.html(
                """
                <script>
                (function() {
                    const doc = window.parent.document;
                    if (doc.__promptHoverTooltipInstalled) return;
                    doc.__promptHoverTooltipInstalled = true;

                    const tooltip = doc.createElement('div');
                    tooltip.id = 'prompt-hover-tooltip';
                    Object.assign(tooltip.style, {
                        position: 'fixed',
                        zIndex: '99999',
                        maxWidth: '900px',
                        minWidth: '560px',
                        maxHeight: '70vh',
                        overflowY: 'auto',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        lineHeight: '1.45',
                        fontSize: '13px',
                        padding: '14px 16px',
                        borderRadius: '10px',
                        border: '1px solid rgba(120,120,120,0.35)',
                        background: 'rgba(255,255,255,0.98)',
                        boxShadow: '0 10px 24px rgba(0,0,0,0.18)',
                        color: '#111',
                        display: 'none',
                        pointerEvents: 'none'
                    });
                    doc.body.appendChild(tooltip);

                    const placeTooltip = (event) => {
                        const gap = 18;
                        const width = tooltip.offsetWidth || 700;
                        const height = tooltip.offsetHeight || 200;
                        let left = event.clientX + gap;
                        let top = event.clientY + gap;

                        if (left + width > window.parent.innerWidth - 12) {
                            left = Math.max(12, event.clientX - width - gap);
                        }
                        if (top + height > window.parent.innerHeight - 12) {
                            top = Math.max(12, window.parent.innerHeight - height - 12);
                        }
                        tooltip.style.left = `${left}px`;
                        tooltip.style.top = `${top}px`;
                    };

                    const bindTextarea = (textarea) => {
                        if (textarea.dataset.hoverTooltipBound === '1') return;
                        textarea.dataset.hoverTooltipBound = '1';

                        textarea.addEventListener('mouseenter', (event) => {
                            const value = (textarea.value || '').trim();
                            if (!value) return;
                            tooltip.textContent = value;
                            tooltip.style.display = 'block';
                            placeTooltip(event);
                        });

                        textarea.addEventListener('mousemove', (event) => {
                            if (tooltip.style.display === 'block') {
                                placeTooltip(event);
                            }
                        });

                        textarea.addEventListener('mouseleave', () => {
                            tooltip.style.display = 'none';
                        });
                    };

                    const bindAll = () => {
                        const textareas = doc.querySelectorAll('[data-testid="stTextArea"] textarea');
                        textareas.forEach(bindTextarea);
                    };

                    bindAll();
                    const observer = new MutationObserver(() => bindAll());
                    observer.observe(doc.body, { childList: true, subtree: true });
                })();
                </script>
                """,
                height=0,
                width=0,
        )


def filter_rows(rows: list[dict], selected_categories: list[str], keyword: str, status_filter: str) -> list[dict]:
    keyword_lower = keyword.strip().lower()
    filtered = []
    for row in rows:
        if selected_categories and row["category"] not in selected_categories:
            continue
        if keyword_lower:
            haystack = "\n".join(
                [
                    row["group_key"],
                    row.get("prompt") or "",
                    " ".join(row.get("edited_attributes") or []),
                ]
            ).lower()
            if keyword_lower not in haystack:
                continue
        if status_filter == "仅完整" and not row["complete"]:
            continue
        if status_filter == "仅缺图" and row["complete"]:
            continue
        filtered.append(row)
    return filtered


def paginate_rows(rows: list[dict], page_size: int, page_number: int) -> tuple[list[dict], int]:
    if not rows:
        return [], 1
    total_pages = max(1, math.ceil(len(rows) / page_size))
    page_number = min(max(page_number, 1), total_pages)
    start = (page_number - 1) * page_size
    end = start + page_size
    return rows[start:end], total_pages


def render_row(record: dict, visible_models: list[str], all_models: list[str], view_scope: str):
    st.markdown(f"### {record['group_key']}")
    prompt_en = (record.get("prompt") or "").strip()
    prompt_zh = (record.get("prompt_zh") or "").strip()

    if prompt_en and prompt_zh:
        prompt_display_text = f"[EN]\n{prompt_en}\n\n[ZH]\n{prompt_zh}"
    elif prompt_en:
        prompt_display_text = f"[EN]\n{prompt_en}"
    elif prompt_zh:
        prompt_display_text = f"[ZH]\n{prompt_zh}"
    else:
        prompt_display_text = "未找到 prompt"

    st.caption("输入 Prompt（英文 / 中文）")
    st.text_area(
        label="输入 Prompt（英文 / 中文）",
        value=prompt_display_text,
        height=160,
        disabled=True,
        key=f"prompt_{view_scope}_{record['group_key']}",
        label_visibility="collapsed",
    )
    st.caption("可拖动文本框右下角来调整高度")

    attrs = record.get("edited_attributes") or []
    if attrs:
        st.caption("编辑属性: " + ", ".join(attrs))

    missing_models = record.get("missing_models") or []
    if missing_models:
        st.caption("缺失结果: " + ", ".join(format_model_label(alias) for alias in missing_models))

    image_items: list[tuple[str, str]] = []
    image_items.append(("原图", record.get("source_image") or ""))
    for model_alias in visible_models:
        image_items.append((format_model_label(model_alias), record["model_images"].get(model_alias) or ""))

    reference_image = record.get("source_image") or ""
    if not reference_image:
        for _, path in image_items:
            if path:
                reference_image = path
                break
    max_per_row = 5
    ref_size = get_image_size(reference_image) if reference_image else None
    if ref_size and ref_size[0] > ref_size[1]:
        max_per_row = 4

    first_row_count = min(max_per_row, len(image_items))
    col_specs = [1] * first_row_count

    for start in range(0, len(image_items), max_per_row):
        chunk = image_items[start : start + max_per_row]
        if len(chunk) == first_row_count:
            cols = st.columns(col_specs)
        else:
            cols = st.columns(col_specs)
            cols = cols[: len(chunk)]
        for col, (label, image_path) in zip(cols, chunk):
            with col:
                st.caption(label)
                if image_path:
                    st.image(image_path, width='stretch')
                else:
                    if label == "原图":
                        st.warning("未找到原图")
                    else:
                        st.info("缺失")

    zip_bytes, file_name = build_download_payload(record, all_models)
    st.download_button(
        "下载这一组",
        data=zip_bytes,
        file_name=file_name,
        mime="application/zip",
        key=f"download_{view_scope}_{record['group_key']}",
        width='stretch',
    )

    # with st.expander("更多信息"):
    #     metadata_path = record.get("metadata_path") or ""
    #     prompt_path = record.get("prompt_path") or ""
    #     analysis_model = record.get("analysis_model") or "-"
    #     analysis_time = record.get("analysis_time_utc") or "-"
    #     st.write({
    #         "analysis_model": analysis_model,
    #         "analysis_time_utc": analysis_time,
    #         "metadata_path": metadata_path,
    #         "prompt_path": prompt_path,
    #     })
    #     structured_prompt = record.get("structured_prompt") or {}
    #     if structured_prompt:
    #         st.json(structured_prompt)

    st.divider()


def main():
    st.set_page_config(page_title="Benchmark Viewer", layout="wide")
    inject_resizable_textarea_css()
    inject_prompt_hover_tooltip_js()
    st.title("图像生成结果展示")
    st.caption("按样本组聚合 metadata、原图和各模型结果图，适合 benchmark 展示与评审。")

    result_dirs = list_result_directories()
    if not result_dirs:
        st.error(f"当前工作区没有找到 {RESULTS_DIR_NAME} 目录。")
        return

    with st.sidebar:
        st.header("控制面板")
        selected_result_name = result_dirs[0].name
        st.caption(f"结果目录: {selected_result_name}")
        if st.button("重新扫描", width='stretch'):
            st.cache_data.clear()
            st.rerun()

    selected_result_dir = next(path for path in result_dirs if path.name == selected_result_name)
    payload = scan_result_directory(str(selected_result_dir))
    rows = payload["rows"]
    model_aliases = payload["model_aliases"]

    render_meta_prompt(rows, selected_result_dir)
    render_stats(compute_stats(rows, model_aliases), model_aliases)

    categories = sorted({row["category"] for row in rows})
    with st.sidebar:
        selected_categories = st.multiselect("分类筛选", categories)
        visible_models = st.multiselect(
            "显示哪些模型",
            model_aliases,
            default=model_aliases,
            format_func=format_model_label,
        )
        status_filter = st.radio("展示状态", ["全部", "仅完整", "仅缺图"], horizontal=False)
        keyword = st.text_input("关键词搜索", placeholder="支持 group id / prompt / edited_attributes")
        page_size = st.selectbox("每页显示", [5, 10, 20, 50], index=3)

    filtered_rows = filter_rows(rows, selected_categories, keyword, status_filter)

    if not filtered_rows:
        st.warning("当前筛选条件下没有数据。")
        return

    total_pages = max(1, math.ceil(len(filtered_rows) / page_size))
    with st.sidebar:
        page_number = st.number_input("页码", min_value=1, max_value=total_pages, value=1, step=1)

    page_rows, total_pages = paginate_rows(filtered_rows, page_size, int(page_number))
    st.caption(f"当前显示 {len(page_rows)} / {len(filtered_rows)} 组，页码 {int(page_number)} / {total_pages}")

    for record in page_rows:
        render_row(
            record,
            visible_models or model_aliases,
            model_aliases,
            selected_result_dir.name,
        )


if __name__ == "__main__":
    main()
