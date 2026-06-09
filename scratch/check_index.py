from modules.search_index import _extract_streamlit_content_tags, build_search_index

tags = _extract_streamlit_content_tags()
print("=== Streamlit content tags - przykladowe pliki ===")
for path, words in sorted(tags.items(), key=lambda x: -len(x[1]))[:6]:
    print(f"  {path}: {len(words)} slow")
    sample = sorted([w for w in words if len(w) > 4])[:10]
    print(f"    -> {sample}")

print()
idx = build_search_index()
print(f"=== Calkowity indeks: {len(idx)} pozycji ===")
for item in idx[:3]:
    ntags = len(item.get("tags", []))
    title = item.get("title", "?")[:50]
    print(f"  [{title}] -> {ntags} tagow")
