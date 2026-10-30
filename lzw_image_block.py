"""Crop an image block, display its RGB matrix, and encode its bytes with LZW."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

from PIL import Image

MAGIC = b"ILZW1"
MAX_BITS = 16


def lzw_encode(data: bytes) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Return LZW codes and dictionary additions (code, prefix, suffix byte)."""
    if not data:
        return [], []
    dictionary = {bytes((value,)): value for value in range(256)}
    next_code = 256
    code_width = 9
    phrase = bytes((data[0],))
    codes: list[int] = []
    additions: list[tuple[int, int, int]] = []

    for value in data[1:]:
        symbol = bytes((value,))
        candidate = phrase + symbol
        if candidate in dictionary:
            phrase = candidate
            continue
        codes.append(dictionary[phrase])
        if next_code < (1 << MAX_BITS):
            dictionary[candidate] = next_code
            additions.append((next_code, dictionary[phrase], value))
            next_code += 1
            if next_code == (1 << code_width) and code_width < MAX_BITS:
                code_width += 1
        phrase = symbol

    codes.append(dictionary[phrase])
    return codes, additions


def pack_codes(codes: list[int]) -> bytes:
    """Pack codes MSB-first using the same growing width as the LZW encoder."""
    if not codes:
        return b""
    output = bytearray()
    buffer = 0
    buffered_bits = 0
    code_width = 9
    next_code = 256
    for index, code in enumerate(codes):
        buffer = (buffer << code_width) | code
        buffered_bits += code_width
        while buffered_bits >= 8:
            buffered_bits -= 8
            output.append((buffer >> buffered_bits) & 0xFF)
        if index < len(codes) - 1 and next_code < (1 << MAX_BITS):
            next_code += 1
            if next_code == (1 << code_width) and code_width < MAX_BITS:
                code_width += 1
    if buffered_bits:
        output.append((buffer << (8 - buffered_bits)) & 0xFF)
    return bytes(output)


def unpack_and_decode(data: bytes, code_count: int) -> bytes:
    """Decode this tool's MSB-first, variable-width LZW code stream."""
    if code_count == 0:
        return b""
    dictionary = {value: bytes((value,)) for value in range(256)}
    code_width = 9
    next_code = 256
    bit_position = 0

    def read_code(width: int) -> int:
        nonlocal bit_position
        if bit_position + width > len(data) * 8:
            raise ValueError("truncated LZW data")
        value = 0
        for _ in range(width):
            byte_index, bit_index = divmod(bit_position, 8)
            value = (value << 1) | ((data[byte_index] >> (7 - bit_index)) & 1)
            bit_position += 1
        return value

    first = read_code(code_width)
    if first not in dictionary:
        raise ValueError("invalid first LZW code")
    phrase = dictionary[first]
    output = bytearray(phrase)
    for _ in range(code_count - 1):
        # The encoder has added one dictionary phrase ahead of the decoder.
        # Grow the read width just before the next code when that phrase
        # brings the decoder to the encoder's width boundary.
        if next_code == (1 << code_width) - 1 and code_width < MAX_BITS:
            code_width += 1
        code = read_code(code_width)
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            entry = phrase + phrase[:1]
        else:
            raise ValueError("invalid LZW code stream")
        output.extend(entry)
        if next_code < (1 << MAX_BITS):
            dictionary[next_code] = phrase + entry[:1]
            next_code += 1
        phrase = entry
    return bytes(output)


def write_matrix_csv(path: Path, pixels: bytes, width: int, height: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for y in range(height):
            row = []
            for x in range(width):
                offset = (y * width + x) * 3
                red, green, blue = pixels[offset : offset + 3]
                row.append(f"#{red:02X}{green:02X}{blue:02X}")
            writer.writerow(row)


def encode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if min(args.x, args.y) < 0 or min(args.width, args.height) <= 0:
        parser.error("x and y must be non-negative; width and height must be positive")
    image_path = args.image.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image_width, image_height = image.size
            if args.x + args.width > image_width or args.y + args.height > image_height:
                parser.error(f"crop exceeds image bounds {image_width}x{image_height}")
            pixels = image.crop((args.x, args.y, args.x + args.width, args.y + args.height)).tobytes()
    except OSError as exc:
        parser.error(f"cannot read image {image_path}: {exc}")

    codes, additions = lzw_encode(pixels)
    packed = pack_codes(codes)
    metadata = {
        "format": "RGB8",
        "width": args.width,
        "height": args.height,
        "raw_bytes": len(pixels),
        "code_count": len(codes),
        "source_image": image_path.name,
        "source_size": [image_width, image_height],
        "rectangle_xywh": [args.x, args.y, args.width, args.height],
        "bit_order": "MSB-first, variable-width LZW, 9 to 16 bits",
    }
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    output_path.write_bytes(MAGIC + struct.pack(">I", len(header)) + header + packed)

    matrix_path = output_path.with_name(output_path.stem + "_matrix.csv")
    dictionary_path = output_path.with_name(output_path.stem + "_dictionary.csv")
    write_matrix_csv(matrix_path, pixels, args.width, args.height)
    with dictionary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("code", "prefix_code", "suffix_byte"))
        writer.writerows(additions)

    print(f"Cropped RGB matrix: {args.width} x {args.height} -> {matrix_path}")
    print(f"LZW dictionary:     {len(additions):,} entries -> {dictionary_path}")
    print(f"LZW codes:          {len(codes):,}")
    print(f"Uncompressed data:  {len(pixels):,} bytes")
    print(f"LZW file:           {output_path} ({output_path.stat().st_size:,} bytes)")
    return 0


def decode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    coded_path = args.decode.expanduser().resolve()
    try:
        blob = coded_path.read_bytes()
        if not blob.startswith(MAGIC) or len(blob) < len(MAGIC) + 4:
            raise ValueError("not an image-block LZW file")
        header_size = struct.unpack_from(">I", blob, len(MAGIC))[0]
        header_start = len(MAGIC) + 4
        metadata = json.loads(blob[header_start : header_start + header_size])
        packed = blob[header_start + header_size :]
        pixels = unpack_and_decode(packed, metadata["code_count"])
        if len(pixels) != metadata["raw_bytes"]:
            raise ValueError("decoded byte count does not match the header")
        image = Image.frombytes("RGB", (metadata["width"], metadata["height"]), pixels)
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, struct.error) as exc:
        parser.error(f"cannot decode {coded_path}: {exc}")
    print(f"Decoded image: {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop an RGB image block and encode it with LZW.")
    parser.add_argument("image", nargs="?", type=Path, help="input image to crop and encode")
    parser.add_argument("--x", type=int, default=0, help="crop left coordinate")
    parser.add_argument("--y", type=int, default=0, help="crop top coordinate")
    parser.add_argument("--width", type=int, help="crop width in pixels")
    parser.add_argument("--height", type=int, help="crop height in pixels")
    parser.add_argument("--output", type=Path, default=Path("image-block.lzw"), help="output LZW or decoded image")
    parser.add_argument("--decode", type=Path, help="decode an .lzw file to an image instead of encoding")
    args = parser.parse_args()
    if args.decode:
        return decode(args, parser)
    if args.image is None or args.width is None or args.height is None:
        parser.error("encoding requires an image plus --width and --height")
    return encode(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
