"""
AI-Based Product Identification from Engineering Drawings
Evaluates: AWS Textract Queries | AWS Rekognition | AWS Bedrock Claude
"""

import boto3
import base64
import json
import time
import io
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from PIL import Image, ImageEnhance, ImageFilter


def _boto3_client(service: str, region: str):
    """Create boto3 client — reads from Streamlit secrets if running on cloud,
    otherwise falls back to local ~/.aws/credentials."""
    try:
        import streamlit as st
        creds = st.secrets.get("aws", {})
        if creds:
            return boto3.client(
                service,
                region_name=creds.get("aws_region", region),
                aws_access_key_id=creds.get("aws_access_key_id"),
                aws_secret_access_key=creds.get("aws_secret_access_key"),
                aws_session_token=creds.get("aws_session_token"),
            )
    except Exception:
        pass
    return boto3.client(service, region_name=region)


def preprocess_image(image_bytes: bytes) -> bytes:
    """
    Enhance image quality before sending to AWS services.
    - Convert to RGB
    - Resize to optimal resolution (max 2800px on longest side)
    - Sharpen to make text clearer
    - Boost contrast so annotations stand out
    - Output as high-quality JPEG under 5MB (AWS limit)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize — keep aspect ratio, max 2800px on longest side
    max_side = 2800
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Sharpen text edges
    img = img.filter(ImageFilter.SHARPEN)

    # Boost contrast slightly so faint annotations are readable
    img = ImageEnhance.Contrast(img).enhance(1.3)

    # Boost sharpness
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    # Save as JPEG, quality 92
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92, optimize=True)
    result = buf.getvalue()

    # If still over 5MB, reduce quality
    if len(result) > 5 * 1024 * 1024:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75, optimize=True)
        result = buf.getvalue()

    return result


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class QueryAnswer:
    """A single Textract query and its extracted answer."""
    alias: str
    question: str
    answer: str
    confidence: float


@dataclass
class TextLine:
    """A line of text detected by any service."""
    text: str
    confidence: float


@dataclass
class ProductMatch:
    name: str
    model_number: str
    confidence: float
    context: str


@dataclass
class ServiceResult:
    service: str
    latency_ms: float
    # Textract Queries populates this
    query_answers: list[QueryAnswer]
    # Rekognition / Bedrock raw text
    text_lines: list[TextLine]
    # Bedrock also returns structured products directly
    products: list[ProductMatch]
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# 1. AWS Textract — Queries (targeted attribute extraction)
# ---------------------------------------------------------------------------

# Define the specific product attributes we want to extract from any drawing.
# Textract will locate the answer for each question anywhere in the image.
PRODUCT_QUERIES = [
    {"Text": "What is the product name?",                                          "Alias": "PRODUCT_NAME"},
    {"Text": "What is the model number or part number?",                           "Alias": "MODEL_NUMBER"},
    {"Text": "What is the brand or manufacturer name?",                            "Alias": "BRAND"},
    {"Text": "What product or material is specified in the notes?",                "Alias": "SPECIFIED_PRODUCT"},
    {"Text": "What equipment or component is being installed?",                    "Alias": "INSTALLED_PRODUCT"},
    {"Text": "What adhesive, epoxy, anchor, or fastener product is referenced?",   "Alias": "FIXING_PRODUCT"},
    {"Text": "What electrical, mechanical, or HVAC product is mentioned?",         "Alias": "MEP_PRODUCT"},
    {"Text": "What material or specification standard is listed?",                 "Alias": "MATERIAL_SPEC"},
]


class TextractQueryExtractor:
    """
    Uses analyze_document with FeatureTypes=['QUERIES'].
    You supply targeted questions; Textract finds the answer directly in the image.
    No post-processing LLM needed — Textract handles both OCR and attribute extraction.
    """

    def __init__(self, region: str = "us-east-1"):
        self.client = _boto3_client("textract", region)

    def extract(self, image_bytes: bytes) -> list[QueryAnswer]:
        response = self.client.analyze_document(
            Document={"Bytes": image_bytes},
            FeatureTypes=["QUERIES"],
            QueriesConfig={"Queries": PRODUCT_QUERIES},
        )

        # Build a lookup: block_id -> block
        blocks_by_id = {b["Id"]: b for b in response.get("Blocks", [])}

        # Build a lookup: alias -> question text
        alias_to_question = {q["Alias"]: q["Text"] for q in PRODUCT_QUERIES}

        answers = []
        for block in response.get("Blocks", []):
            if block["BlockType"] != "QUERY":
                continue

            alias = block.get("Query", {}).get("Alias", "")
            question = alias_to_question.get(alias, block.get("Query", {}).get("Text", ""))

            # Follow ANSWER relationships to find the QUERY_RESULT block
            answer_text = "NOT FOUND"
            confidence = 0.0
            for rel in block.get("Relationships", []):
                if rel["Type"] == "ANSWER":
                    for answer_id in rel["Ids"]:
                        result_block = blocks_by_id.get(answer_id, {})
                        if result_block.get("BlockType") == "QUERY_RESULT":
                            answer_text = result_block.get("Text", "NOT FOUND")
                            confidence = round(result_block.get("Confidence", 0.0), 2)

            answers.append(QueryAnswer(
                alias=alias,
                question=question,
                answer=answer_text,
                confidence=confidence,
            ))

        return answers


# ---------------------------------------------------------------------------
# 2. AWS Rekognition — detect_text (raw scene-text OCR)
# ---------------------------------------------------------------------------

class RekognitionExtractor:
    """
    Uses detect_text to find all text in the image.
    Returns LINE-level detections — no attribute targeting, just raw text.
    A post-processing step (or manual review) is needed to find products.
    """

    def __init__(self, region: str = "us-east-1"):
        self.client = _boto3_client("rekognition", region)

    def extract(self, image_bytes: bytes) -> list[TextLine]:
        response = self.client.detect_text(Image={"Bytes": image_bytes})
        lines = []
        for det in response.get("TextDetections", []):
            if det["Type"] == "LINE":
                lines.append(TextLine(
                    text=det["DetectedText"],
                    confidence=round(det["Confidence"], 2),
                ))
        return lines


# ---------------------------------------------------------------------------
# 3. AWS Bedrock — Claude vision (end-to-end understanding)
# ---------------------------------------------------------------------------

_SYSTEM = """You are an expert at reading all types of engineering drawings,
construction details, mechanical diagrams, electrical schematics, HVAC plans,
structural drawings, and technical diagrams.

Your job is to:
1. Extract every piece of text visible in the image.
2. Identify ALL product references — this includes any of the following:
   - Adhesives, epoxies, anchors, fasteners
   - Mechanical components (pumps, valves, motors, bearings)
   - Electrical components (cables, conduits, panels, switches)
   - HVAC equipment (ducts, fans, dampers, chillers)
   - Structural materials (beams, plates, bolts, rebar)
   - Coatings, sealants, waterproofing products
   - Any item with a brand name, model number, part number, or spec code
   - Any item referenced by a manufacturer name + product code

A product reference can appear ANYWHERE in the drawing — in notes, callouts,
title blocks, revision tables, legends, or annotation bubbles."""

_USER = """Carefully analyse this engineering drawing image.

Extract ALL visible text and identify EVERY product reference.

Return ONLY valid JSON — no markdown, no explanation:
{
  "all_text": ["every text string visible anywhere in the image"],
  "products": [
    {
      "name": "full product name including product type, e.g. HIT-HY 200-A V3 Adhesive anchor / 3M Fire Barrier Sealant / Hilti KB-TZ2 Expansion Anchor",
      "model_number": "model code or part number only, e.g. HIT-HY 200-A V3 / KB-TZ2 / CP 601S",
      "brand": "manufacturer or brand name, e.g. Hilti / 3M / Sika / Simpson Strong-Tie",
      "confidence": 0.0,
      "context": "the exact annotation or note in the drawing that contains this product reference"
    }
  ]
}

Rules:
- If no products are found, return an empty products list.
- Do not guess — only return products explicitly visible in the drawing.
- A product must have at least a model number OR a brand name to be included.
- If the same product appears multiple times with different abbreviations or
  partial names (e.g. HIT HY-200 and HIT-HY 200-A V3), treat them as ONE product
  and return only the most complete and specific version.
- Return each unique physical product only ONCE — no duplicates."""


class BedrockExtractor:
    """
    Sends the image directly to Claude on Bedrock (multimodal).
    Claude reads the drawing, understands engineering context, and returns
    structured product data — no separate OCR step required.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    ):
        self.client = _boto3_client("bedrock-runtime", region)
        self.model_id = model_id

    def extract(self, image_bytes: bytes, media_type: str = "image/jpeg") -> dict:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "system": _SYSTEM,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(image_bytes).decode(),
                            },
                        },
                        {"type": "text", "text": _USER},
                    ],
                }
            ],
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        raw = json.loads(response["body"].read())["content"][0]["text"]

        # Strip accidental markdown code fences
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _deduplicate_products(products: list[ProductMatch]) -> list[ProductMatch]:
    """
    When the same physical product appears with different levels of specificity
    (e.g. 'HIT HY-200' and 'HIT-HY 200-A V3'), keep only the most complete version.
    Strategy: normalize model numbers to alphanumeric-only uppercase; if one is a
    substring of another, the shorter one is a less-specific duplicate — drop it.
    """
    import re

    if len(products) <= 1:
        return products

    def norm(s: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", s.upper())

    keep = []
    for i, p in enumerate(products):
        ni = norm(p.model_number) or norm(p.name)
        dominated = False
        for j, q in enumerate(products):
            if i == j:
                continue
            nj = norm(q.model_number) or norm(q.name)
            # p is a less-specific duplicate if its key is a strict substring of q's key
            if ni and nj and ni in nj and ni != nj:
                dominated = True
                break
        if not dominated:
            keep.append(p)

    return keep if keep else [max(products, key=lambda p: len(p.model_number))]


def _media_type(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/jpeg")


class ProductExtractor:
    """Runs all three services and prints a side-by-side evaluation report."""

    def __init__(
        self,
        region: str = "us-east-1",
        bedrock_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ):
        self.textract    = TextractQueryExtractor(region)
        self.rekognition = RekognitionExtractor(region)
        self.bedrock     = BedrockExtractor(region, bedrock_model)

    # ------------------------------------------------------------------ #

    def _run_textract(self, image_bytes: bytes) -> ServiceResult:
        t0 = time.time()
        try:
            image_bytes = preprocess_image(image_bytes)
            answers = self.textract.extract(image_bytes)
            # Derive product matches from non-empty answers
            products = [
                ProductMatch(
                    name=a.answer,
                    model_number=a.answer,
                    confidence=a.confidence / 100,
                    context=f"{a.alias}: {a.answer}",
                )
                for a in answers
                if a.answer != "NOT FOUND"
            ]
            return ServiceResult(
                service="AWS Textract — Queries",
                latency_ms=round((time.time() - t0) * 1000, 1),
                query_answers=answers,
                text_lines=[],
                products=products,
            )
        except Exception as exc:
            return ServiceResult(
                service="AWS Textract — Queries",
                latency_ms=round((time.time() - t0) * 1000, 1),
                query_answers=[], text_lines=[], products=[],
                error=str(exc),
            )

    def _run_rekognition(self, image_bytes: bytes) -> ServiceResult:
        t0 = time.time()
        try:
            image_bytes = preprocess_image(image_bytes)
            lines = self.rekognition.extract(image_bytes)
            return ServiceResult(
                service="AWS Rekognition — detect_text",
                latency_ms=round((time.time() - t0) * 1000, 1),
                query_answers=[],
                text_lines=lines,
                products=[],   # raw text only — no product ID without post-processing
            )
        except Exception as exc:
            return ServiceResult(
                service="AWS Rekognition — detect_text",
                latency_ms=round((time.time() - t0) * 1000, 1),
                query_answers=[], text_lines=[], products=[],
                error=str(exc),
            )

    def _run_bedrock(self, image_bytes: bytes, media_type: str) -> ServiceResult:
        t0 = time.time()
        try:
            image_bytes = preprocess_image(image_bytes)
            result = self.bedrock.extract(image_bytes, "image/jpeg")
            text_lines = [
                TextLine(text=t, confidence=1.0)
                for t in result.get("all_text", [])
            ]
            products = _deduplicate_products([
                ProductMatch(
                    name=p.get("name", p.get("model_number", "")),
                    model_number=p.get("model_number", ""),
                    confidence=float(p.get("confidence", 0.0)),
                    context=p.get("context", ""),
                )
                for p in result.get("products", [])
            ])
            return ServiceResult(
                service="AWS Bedrock — Claude vision",
                latency_ms=round((time.time() - t0) * 1000, 1),
                query_answers=[],
                text_lines=text_lines,
                products=products,
            )
        except Exception as exc:
            return ServiceResult(
                service="AWS Bedrock — Claude vision",
                latency_ms=round((time.time() - t0) * 1000, 1),
                query_answers=[], text_lines=[], products=[],
                error=str(exc),
            )

    # ------------------------------------------------------------------ #

    def evaluate(self, image_path: str) -> dict[str, ServiceResult]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(image_path)

        image_bytes = path.read_bytes()
        mt = _media_type(path)

        print(f"\n{'='*65}")
        print(f"  Image : {path.name}  ({len(image_bytes)//1024} KB)")
        print(f"{'='*65}")

        results: dict[str, ServiceResult] = {}

        print("\n[1/3] AWS Textract — Queries ...")
        results["textract"] = self._run_textract(image_bytes)
        _print_status(results["textract"])

        print("\n[2/3] AWS Rekognition — detect_text ...")
        results["rekognition"] = self._run_rekognition(image_bytes)
        _print_status(results["rekognition"])

        print("\n[3/3] AWS Bedrock — Claude vision ...")
        results["bedrock"] = self._run_bedrock(image_bytes, mt)
        _print_status(results["bedrock"])

        return results

    def print_report(self, results: dict[str, ServiceResult]) -> None:
        print(f"\n{'='*65}")
        print("  DETAILED RESULTS")
        print(f"{'='*65}")

        # ---------- Textract ----------
        r = results["textract"]
        print(f"\n{'─'*65}")
        print(f"  SERVICE : {r.service}   [{r.latency_ms} ms]")
        if r.error:
            print(f"  ERROR : {r.error}")
        else:
            print(f"  Queried {len(r.query_answers)} attributes:\n")
            for qa in r.query_answers:
                found = qa.answer != "NOT FOUND"
                flag  = "✔" if found else "✘"
                conf  = f"  ({qa.confidence}%)" if found else ""
                print(f"    {flag}  {qa.alias:<22}  →  {qa.answer}{conf}")

        # ---------- Rekognition ----------
        r = results["rekognition"]
        print(f"\n{'─'*65}")
        print(f"  SERVICE : {r.service}   [{r.latency_ms} ms]")
        if r.error:
            print(f"  ERROR : {r.error}")
        else:
            print(f"  {len(r.text_lines)} text lines detected (manual product filtering needed):\n")
            for line in r.text_lines:
                print(f"    [{line.confidence:5.1f}%]  {line.text}")

        # ---------- Bedrock ----------
        r = results["bedrock"]
        print(f"\n{'─'*65}")
        print(f"  SERVICE : {r.service}   [{r.latency_ms} ms]")
        if r.error:
            print(f"  ERROR : {r.error}")
        else:
            print(f"  {len(r.text_lines)} text items extracted")
            print(f"  {len(r.products)} products identified:\n")
            for p in r.products:
                print(f"    • {p.name}  |  {p.model_number}  |  conf {p.confidence:.2f}")
                print(f"      \"{p.context}\"")

        # ---------- Summary ----------
        print(f"\n{'='*65}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*65}")
        print(f"  {'Service':<35} {'Output':<20} {'Latency':>10}")
        print(f"  {'─'*35} {'─'*20} {'─'*10}")

        for r in results.values():
            if r.error:
                output = "ERROR"
            elif r.query_answers:
                found = sum(1 for qa in r.query_answers if qa.answer != "NOT FOUND")
                output = f"{found}/{len(r.query_answers)} attrs found"
            elif r.products:
                output = f"{len(r.products)} products"
            else:
                output = f"{len(r.text_lines)} text lines (raw)"

            print(f"  {r.service:<35} {output:<20} {r.latency_ms:>8.0f} ms")

        print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_status(r: ServiceResult) -> None:
    if r.error:
        print(f"    ERROR: {r.error}")
    elif r.query_answers:
        found = sum(1 for qa in r.query_answers if qa.answer != "NOT FOUND")
        print(f"    OK — {found}/{len(r.query_answers)} attributes answered  |  {r.latency_ms} ms")
    elif r.text_lines:
        print(f"    OK — {len(r.text_lines)} text lines  |  {r.latency_ms} ms")
    else:
        print(f"    OK — {len(r.products)} products  |  {r.latency_ms} ms")
