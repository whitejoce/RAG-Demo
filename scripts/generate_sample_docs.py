from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PDF_PATH = RAW_DIR / "光缆套餐介绍.pdf"


def build_pdf_bytes() -> bytes:
    lines = [
        "BT",
        "/F1 18 Tf",
        "50 790 Td",
        "(Enterprise Fiber Package Overview) Tj",
        "/F1 12 Tf",
        "0 -28 Td",
        "(1. 100M, 300M, and 1000M packages for branch offices.) Tj",
        "0 -18 Td",
        "(2. Free installation survey and standard wiring guidance.) Tj",
        "0 -18 Td",
        "(3. Optional static IP, SLA, and 7x24 support.) Tj",
        "0 -18 Td",
        "(4. Recommended for video conferencing and ERP access.) Tj",
        "ET",
    ]
    content = "\n".join(lines).encode("latin-1")
    objects: list[bytes] = []

    def add_object(body: bytes) -> None:
        objects.append(body)

    add_object(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    add_object(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    add_object(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
    )
    add_object(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    add_object(
        b"5 0 obj << /Length "
        + str(len(content)).encode("ascii")
        + b" >> stream\n"
        + content
        + b"\nendstream endobj\n"
    )

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            "trailer << /Size {size} /Root 1 0 R >>\n"
            "startxref\n{startxref}\n%%EOF\n"
        ).format(size=len(offsets), startxref=xref_start).encode("ascii")
    )
    return bytes(pdf)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PDF_PATH.write_bytes(build_pdf_bytes())


if __name__ == "__main__":
    main()

