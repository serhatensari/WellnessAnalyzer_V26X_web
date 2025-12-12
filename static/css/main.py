import os
import json
import re
from typing import List, Literal, Dict, Any
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PyPDF2 import PdfReader
from openai import OpenAI

# ====================================================
#  JSON MODE'DAN GELEN BİRDEN FAZLA PARÇAYI BİRLEŞTİRME
# ====================================================

def merge_json_mode_payloads(raw_json_list: List[str]) -> Dict[str, Any]:
    """
    JSON MODE'dan gelen birden fazla ham JSON stringini tek bir analysis dict içinde birleştirir.
    - 1. JSON: kisi_bilgileri, vucut_formu, genel_bulgu, plan vs.
    - 2. JSON: sistem_kartlari
    """
    merged: Dict[str, Any] = {}
    system_cards: List[Dict[str, Any]] = []

    for raw in raw_json_list:
        raw = (raw or "").strip()
        if not raw:
            continue

        try:
            obj = json.loads(raw)
        except Exception as e:
            print("JSON parse hatası:", e, "ham:", raw[:200])
            continue

        # sistem_kartlari geldiyse topla
        if "sistem_kartlari" in obj:
            # güvenli olsun diye liste bekliyoruz
            if isinstance(obj["sistem_kartlari"], list):
                system_cards = obj["sistem_kartlari"]

        # diğer alanlar (kisi_bilgileri, genel_bulgu vs.)
        for k, v in obj.items():
            if k == "sistem_kartlari":
                continue
            merged[k] = v

    if system_cards:
        merged["sistem_kartlari"] = system_cards

    return merged


# ====================================================
#  ÜRÜN / CİNSİYET FİLTRE YARDIMCILARI
# ====================================================

def normalize_gender(g: str) -> str:
    if not g:
        return "unisex"
    g = g.strip().lower()
    if g.startswith("e") or g.startswith("m"):  # "erkek" / "male"
        return "male"
    if g.startswith("d") or g.startswith("k") or g.startswith("f"):  # "dişi"/"kadın"/"female"
        return "female"
    return "unisex"


def normalize_name_for_compare(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


# (Eski tek marka örnek yapıyı sadece legacy filtre için ayrı tuttum)
LEGACY_BRAND_PRODUCTS = {
    "XAura Global": [
        {"name": "XAura X-Omega", "gender": "unisex"},
        {"name": "XAura X-Dk (Vitamin D & K)", "gender": "unisex"},
        {"name": "XAura X-12 (B Vitamini Kompleksi)", "gender": "unisex"},
        {"name": "XAura FemmeBalance", "gender": "female"},
        {"name": "XAura MaleVital", "gender": "male"},
        {"name": "XAura X-Recovery", "gender": "unisex"},
        {"name": "XAura Beauty Collagen Mask", "gender": "female"},
    ],
    "OneMore International": [
        {"name": "OneMore Core Omega", "gender": "unisex"},
        {"name": "OneMore Women's Balance", "gender": "female"},
        {"name": "OneMore Men's Vital", "gender": "male"},
    ],
}

def load_brand_products_map_legacy() -> dict:
    """Sadece filter_onemore_products_in_analysis için; şu an aktif olarak kullanılmıyor."""
    return LEGACY_BRAND_PRODUCTS


def product_allowed_for(brand_products_map: dict, brand_label: str, product_name: str, person_gender: str) -> bool:
    if not product_name:
        return False
    brand_list = brand_products_map.get(brand_label, []) or []
    pname_norm = normalize_name_for_compare(product_name)
    person_gender_norm = normalize_gender(person_gender)

    for p in brand_list:
        if normalize_name_for_compare(p.get("name", "")) == pname_norm:
            p_gender = (p.get("gender", "unisex") or "unisex").strip().lower()
            if p_gender in ("unisex", person_gender_norm):
                return True
            else:
                return False
    return False


def filter_onemore_products_in_analysis(analysis: dict, brand_label: str, person_gender: str) -> dict:
    """
    Eski uyumluluk için bırakıldı; asıl kullanılan sistem apply_brand_product_filter.
    İstersen tamamen kaldırabilirsin.
    """
    brand_products_map = load_brand_products_map_legacy()
    pg = person_gender or ""

    def filter_list(lst):
        newlst = []
        for item in lst or []:
            if not isinstance(item, dict):
                continue
            name = item.get("urun") or item.get("urun_adi") or item.get("name") or item.get("title") or ""
            if product_allowed_for(brand_products_map, brand_label, name, pg):
                newlst.append(item)
        return newlst

    if isinstance(analysis.get("genel_bulgu"), dict):
        analysis["genel_bulgu"]["onemore_urun_onerileri"] = filter_list(
            analysis["genel_bulgu"].get("onemore_urun_onerileri", [])
        )

    for key in ("kanallar_ve_kollateraller_detay", "insan_bilinc_duzeyi_detay"):
        if isinstance(analysis.get(key), dict):
            analysis[key]["onemore_urun_onerileri"] = filter_list(
                analysis[key].get("onemore_urun_onerileri", [])
            )

    if isinstance(analysis.get("sistem_kartlari"), list):
        for card in analysis["sistem_kartlari"]:
            if isinstance(card, dict):
                card["urun_onerileri"] = filter_list(card.get("urun_onerileri", []))

    return analysis


# ====================================================
#  OPENAI KURULUM
# ====================================================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = "gpt-4.1-mini"

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DetailLevel = Literal["ultra"]

DETAIL_LEVELS: Dict[str, str] = {
    "ultra": "Ultra Detaylı (En kapsamlı)",
}

SYSTEM_NAMES: List[str] = [
    "1. Kardiyovasküler ve Serebrovasküler",
    "2. Gastrointestinal Fonksiyon",
    "3. Kalın Bağırsak Fonksiyonu",
    "4. Karaciğer Fonksiyonu",
    "5. Safra Kesesi Fonksiyonu",
    "6. Pankreas Fonksiyonu",
    "7. Böbrek Fonksiyonu",
    "8. Akciğer Fonksiyonu",
    "9. Beyin Sinir Sistemi",
    "10. Kemik Hastalığı",
    "11. Kemik Mineral Yoğunluğu",
    "12. Romatoid Kemik Hastalığı",
    "13. Kan Şekeri",
    "14. İz Elementler",
    "15. Vitaminler",
    "16. Amino Asitler",
    "17. Koenzimler",
    "18. Endokrin Fonksiyon",
    "19. Esansiyel Yağ Asidi",
    "20. Endokrin Sistem",
    "21. Bağışıklık Sistemi",
    "22. Tiroid",
    "23. İnsan Toksini",
    "24. Ağır Metal",
    "25. Temel Fiziksel Kalite",
    "26. Alerji",
    "27. Obezite",
    "28. Cilt",
    "29. Göz",
    "30. Kolajen",
    "31. Kanallar ve kollateraller",
    "32. Kalp ve beyin nabzı",
    "33. Beyin dalgası",
    "34. Kan lipidi",
    "35. Meme (kadın)",
    "36. Adet döngüsü (kadın)",
    "37. Kadınlık Hormonu (kadın)",
    "38. Prostat (erkek)",
    "39. Erkek Cinsel Fonksiyonu (erkek)",
    "40. Sperm ve meni (erkek)",
    "41. Erkek Hormonu (erkek)",
    "42. İnsan Bağışıklığı",
    "43. İnsan Bilinç Düzeyi",
    "44. Solunum Fonksiyonu",
    "45. Kemik Gelişimi (çocuk)",
    "46. Çocukluk Sağlık Durumu",
]

# ============================================================
#  TÜM MARKALARIN ÜRÜN LİSTESİ (KOPYALA-YAPIŞTIR)
# ============================================================

PRODUCT_LISTS = {

    # --------------------------------------------------------
    # OneMore International
    # --------------------------------------------------------
    "onemore": [
        "Painless night glu",
        "Slim style",
        "B12 plus",
        "Dekamin",
        "Melatonin plus",
        "Omevia",
        "Night Ladies",
        "Night Gentlemen",
        "Sornie Collagen Patch",
        "Sornie ANTI - AGING MASK",
        "LUMIERE SUN SCREEN SPF50+",
        "GLUTAMORE",
        "OMICOFF CLASSIC",
        "OMICOFF LATTE",
        "OMICOFF MOCHA",
        "HAIR REPAIR – ŞAMPUAN",
        "HAIR REPAIR MASK",
        "LUMIÈRE SHOWER GEL",
        "PAİNLESS NİGHT GLU PLUS+",
        "FİTMORE SHAKE"
    ],

    # --------------------------------------------------------
    # Xaura Global
    # --------------------------------------------------------
    "xaura": [
        "XAura Pain",
        "XAura X-Col",
        "XAura X-Recovery",
        "XAura X-She",
        "XAura X-He",
        "XAura X-Omega",
        "XAura X-Night",
        "XAura X-12",
        "XAura X-Dk",
        "XAura X-Lim",
        "Xaura Coffee Black Reishi",
        "Xaura 2 in 1 Coffee Reishi",
        "Xaura Coffee Latte Reishi",
        "Xaura Beauty Sun Cream",
        "XAura Beauty Collagen Mask",
        "XAura Beauty Aloe Mask",
        "XAura Beauty Eye Cream",
        "XAura Beauty Collagen Soothing Gel",
        "XAura Beauty Rice Mask",
        "XAura Beauty Seaweed Mask",
        "XAura Beauty BB Cream",
        "XAura Beauty Foam Cleanser",
        "XAura Beauty Whitening Cream"
    ],

    # =========================
    # Amare Global
    # =========================
    "amare": [
        "Amare Sunrise",
        "Amare Sunset",
        "Amare Nitro Plus",
        "HL5 Kolajen Protein",
        "FIT20 Whey Protein",
        "RESTORE",
        "ON SHOTS",
        "ORIGIN",
        "R-STOʊR",
        "EDGE",
        "NRGI",
        "MNTA",
        "IGNT HER",
        "IGNT HIM",
        "Wellness Üçgeni Seti",
    ],

    # =========================
    # Forever Living
    # =========================
    "forever": [
        "Eco 9 Vanilla",
        "Eco 9 Chocolate",
        "C9 Forever Lite Ultr Chocolate Pouch",
        "C9 Forever Lite Ultr Vanilla Pouch",
        "F15 Beginner Ultr Chocolate Pouch",
        "Clean C9 - Forever Lite Ultr Vanilla",
        "Clean C9 - Forever Lite Ultr Chocolate",
        "Start your Journey pack Chocalate",
        "My Fit 1",
        "My Fit 2",
        "My Fit 3",
        "My Fit 4",
        "My Fit 11",
        "My Fit 6",
        "Aloe Vera Gel",
        "Aloe Lips",
        "Forever Bee Pollen",
        "Forever Bee Propolis",
        "Forever Bright Toothgel",
        "Aloe Berry Nectar",
        "Forever Royal Jelly",
        "Nature Min",
        "Arctic-Sea Omega-3",
        "Aloe First Spray",
        "Absorbent-C",
        "Aloe Propolis Creme",
        "Aloe Vera Gelly",
        "Aloe Moisturizing Lotion",
        "Aloe Heat Lotion",
        "Forever Garlic Thyme",
        "Aloe Ever Shield Stick Deo",
        "R-3 Factor Skin Defense Creme",
        "Gentleman’s Pride",
        "Forever Garcinia Plus",
        "Lycium Plus",
        "Forever Aloe Bits N’Peaches",
        "Forever Alpha E Factor",
        "Forever B12 Plus",
        "Forever Freedom",
        "Aloe Blossom Herbal Tea",
        "Aloe MSM Gel",
        "Forever Calcium",
        "Forever Bee Honey",
        "Forever Aloe Scrub",
        "Forever Active HA",
        "Avocado Face & Body Soap",
        "Forever Leane",
        "Sonya Deep Moisturizing Cream",
        "Forever NutraQ10",
        "Mask Powder",
        "Forever Immublend",
        "Vitolize Women’s",
        "Forever Daily",
        "Infinite Kit",
        "Forever Therm",
        "Forever Fiber",
        "Forever Lite Ultr Vanilla - Pouch",
        "Forever Lite Ultr Chocolate - Pouch",
        "Forever ARGI+ - Pouch",
        "Infinite Hydrating Cleanser",
        "Infinite Firming Serum",
        "Infinite Restoring Cream",
        "Smoothing Exfoliator",
        "Balancing Toner",
        "Awakening Eye Cream",
        "Aloe Cooling Lotion",
        "Forever Immune Gummy",
        "Sonya Refreshing Gel Cleanser",
        "Sonya illuminating Gel",
        "Sonya Refining Gel Mask",
        "Sonya Soothing Gel Moisturizer",
        "Sonya Daily Skincare Kit",
        "Forever Active PRO B",
        "Aloe Activator",
        "Forever Marine Collagen",
        "Forever Bio-Cellulose Mask",
        "Aloe Sunscreen-New",
        "Hydrating Serum",
        "Forever i-VSN",
        "Aloe Liquid Soap",
        "Aloe Jojoba Shampoo",
        "Aloe Jojoba Conditioner",
        "Nourishing Hair Oil",
        "Forever Alofa Fine Fragrance",
        "Forever Malosi Fine Fragrance",
        "Protecting Day Lotion",
        "Aloe Body Wash",
        "Aloe Body Lotion",
        "Deep Moisturizing Cream",
        "Tightening Mask Powder",
        "Replenishing Skin Oil",
        "Forever Plant Protein",
        "Forever Bio-Cellulose Mask",
        "Forever Sensatiable",
        "Logic Aloe Gel Cleanser",
        "Logic Gel Balancing Essence",
        "Logic Aloe Soothing Gel Moisturizer",
        "Forever Absorbent-D",
        "Forever Aloe Mango",
        "Aloe Vera Gel (330 ml Pack of-3)",
        "Aloe Berry Nectar (330 ml Pack of-3)",
        "Forever Peaches (330 ml Pack of-3)",
        "Aloe Vera Gel (330 ml Pack of-12)",
        "Aloe Berry Nectar (330 ml Pack of-12)",
        "Forever Peaches (330 ml Pack of-12)",
    ],

    # --------------------------------------------------------
    # VIP International
    # --------------------------------------------------------
    "vip_international": [
        "Vip Epimedyum & Çakşır",
        "VIPA (Gençlik İksiri) Kolajen",
        "Multi Vitamin Toz İçecek",
        "Stop Mıx",
        "VIP Kapsül",
        "Kımıs Tablet",
        "Niksir Tablet",
        "OxiVIP Tablet",
        "Moringa Tablet",
        "Çörek Otu Kapsül",
        "Nitro Toz",
        "Nitro Tablet",
        "Moringalı Kahve (Sade)",
        "Moringalı Kahve (Karışık)",
        "Çörek Otlu Kahve",
        "Moringalı Form Çay",
        "Reishi Mantarlı Kahve",
        "Kolloidal Gümüş Suyu",
        "Kolloidal Altın Suyu",
        "Kolloidal Magnezyum Suyu",
        "Gold Power Strips Bant",
        "Tip2 Power Kolajen Bant",
        "Germanyum & Gümüş İyonlu Bant",
        "Heat Power Krem",
        "Gümüş İyonlu Krem",
        "Altın iyonlu Krem",
        "İntim Yıkama Jeli",
        "Ganodermalı Sabun",
        "Gümüş İyonlu Şampuan",
        "Altın iyonlu Şampuan",
        "Altın iyonlu Yüz Yıkama Jeli",
        "Altın iyonlu Maske",
        "Gümüş İyonlu Sabun",
        "Moringalı Sabun",
        "Gümüş iyonlu Diş Macunu"
    ],

    # =========================
    # Som International
    # =========================
    "som": [
        "Som Power",
        "Som King",
        "Som Queen",
        "Som Kids",
        "Som Force",
        "Best Fell",
        "7 Plus 21 Detox",
        "Som Mag Plus",
        "Ionic Water",
        "Slim Form Tea",
        "Active Pro",
        "Som Mist",
        "Collagen Yüz Maskesi Sıkılaştırıcı Kırışıklık Karşıtı",
        "Vitamin C Nemlendirici ve Onarıcı Yüz Kremi",
        "Gentian Eye Kremi",
        "Deep Vita C Kapsül Yüz Kremi",
        "Revive Serum",
        "Glow Deep Serum",
        "Glow Anti Aging Serum",
        "Calming Serum",
        "Som Mist (Gençlik İksiri)",
        "Anti Aging Krem",
        "Som Coffee (Active Pro Anti Aging)",
        "Power Focus Coffee",
        "Som Shampoo",
    ],

    # =========================
    # Onyedi Wellness
    # =========================
    "onyedi": [
        "Onyedi Kolajen",
        "Onyedi K-Patch Ginseng Ağrı Bandı",
        "Onyedi Classic Coffee Reishi Kahve",
        "Onyedi Mocha Coffee Reishi Kahve",
        "Onyedi Latte Coffee Reishi Latte",
        "Onyedi Hot Chocolate Coffee Reishi Sıcak Çikolata",
    ],

    # --------------------------------------------------------
    # Algo / Algophyco TTS
    # --------------------------------------------------------
    "algo": [
        "Power Patch",
        "XXL Algo Ginseng 365",
        "Omega & Koenzim Patch",
        "Smart Patch",
        "Pineal Patch",
        "Queen Royal Jelly",
        "Bitter Melon",
        "Man Patch",
        "Woman Patch",
        "Fit Patch",
        "Lipovit",
        "Vit B Pro",
        "Neurovit",
        "Arginine Plus",
        "Outovit",
        "Algofit",
        "Algo Mega",
        "Vitoma",
        "Protamin",
        "Radix Pro",
        "Starmoon Strong Coffee",
        "Liveliness Hydrogel Mask",
        "Liveliness Güneş Koruyucu 50+ SPF"
    ],

    # --------------------------------------------------------
    # Atomy
    # --------------------------------------------------------
    "atomy": [
        "Atomy Color Food Vitamin C",
        "Atomy Noni Pouch",
        "Atomy Probiotics 10+ Plus",
        "Atomy Spirulina",
        "Atomy Psyllium Husk",
        "Atomy Hongsamdan Ginseng",
        "Atomy Rhodiola Milk Thistle",
        "Atomy Pu'er Tea",
        "Atomy Lutein",
        "Atomy Vitamin B Complex",
        "Atomy Lactium",
        "Atomy Vegetable Algae Omega 3",
        "Atomy Cafe Arabica",
        "Atomy Cafe Arabica Black",
        "Atomy Scalpcare Şampuan",
        "Atomy Scalpcare Saç Kremi",
        "Atomy Hair Oil Complex",
        "Atomy Saengmodan Saç Toniği",
        "Atomy Vücut Losyonu",
        "Atomy Rich Body Duş Jeli",
        "Atomy Marine Göz Bandı",
        "Atomy Marine Jel Maske",
        "Atomy Absolute Essence UV Güneş Koruyucu",
        "Atomy Evening Care Set",
        "Atomy Absolute CellActive Cilt Bakım Seti",
        "Atomy Derma Real Cica Set",
        "Atomy Homme Cilt Bakım Seti",
        "Atomy THE FAME Cilt Bakım Seti"
    ],

    # --------------------------------------------------------
    # Doctorem International
    # --------------------------------------------------------
    "doctorem": [
        "Gin Plus",
        "Vita Plus",
        "Body Plus",
        "Epifiz Plus",
        "Omega Plus",
        "Man Plus",
        "Woman Plus",
        "Thin Plus Normal",
        "Thin Plus Belly",
        "Vita Lina Şampuan",
        "Aura Nova Temizleme Köpüğü",
        "Aura Nova BB Cream",
        "Aura Nova Anti Aging Krem",
        "Aura Nova Anti Aging Serum",
        "Aura Nova OXYMASK",
        "Aura Nova Anti Aging Tonik Lotion",
        "Aura Lena Maske Seti",
        "Aura Mina (İntim Jel)",
        "Aura Sole (Manyetik Tabanlık)"
    ],

    # =========================
    # İndeva Global
    # =========================
    "indeva": [
        "Bye Pain",
        "Bye Suger",
        "Bye Stress",
        "Bye Lack",
        "Bye Nox",
        "Mary Saç Bakım Şampuanı",
        "Mary Saç Bakım Serumu",
        "Mary Sakal Bakım Serumu",
        "Mary Dry Oil",
        "Mary Hyaluronik Asit Duş Jeli",
        "Mary Tea Tree Duş Jeli (Çay ağacı & Ylang Ylang)",
        "Mary Nemlendirici Duş Jeli (Çilek & Hindistan Cevizi)",
        "Mary Nemlendirici Duş Jeli (Ahududu & Meyve Özleri)",
        "Mary Ferahlatıcı Erkek Duş Jeli (Bergamot & Yasemin)",
        "Mary Body Scrub",
        "Mary Prufresh Yüz Yıkama Jeli",
        "Mary Revitaluxe Anti Aging Serum",
        "Mary Liftension Collagen Serum",
        "Mary Liftension Collagen Complex Krem",
        "Mary Hydro 3D Krem",
        "Cilt Bakımı 5’li Set",
    ],

    # =========================
    # Now International
    # =========================
    "now": [
        "Pain End",
        "Hex Now",
        "Mig Ver",
        "Vit Now",
        "Sleep",
        "TEAMAZING",
        "REDUCE",
        "PRO TEA",
        "Fit Now",
        "Energy",
        "STAY UP",
        "Hair Shiny Saç Serumu",
        "Now Sun Time",
        "Golden Collagen Maske",
        "Vit D3 K2",
        "Now Classic Reishi Mantar",
        "CBD Oil",
        "Now Glow CBD + Collagen Jel",
        "NOW PLUS 5.0 CBD Oil",
        "Hermona Plus Oral Sprey",
    ],

    # Aynı listeyi olası lower-case kullanım için de ekliyorum
    "now": [
        "Pain End",
        "Hex Now",
        "Mig Ver",
        "Vit Now",
        "Sleep",
        "TEAMAZING",
        "REDUCE",
        "PRO TEA",
        "Fit Now",
        "Energy",
        "STAY UP",
        "Hair Shiny Saç Serumu",
        "Now Sun Time",
        "Golden Collagen Maske",
        "Vit D3 K2",
        "Now Classic Reishi Mantar",
        "CBD Oil",
        "Now Glow CBD + Collagen Jel",
        "NOW PLUS 5.0 CBD Oil",
        "Hermona Plus Oral Sprey",
    ],

    # =========================
    # Oxo Global
    # =========================
    "oxo": [
        "LaGrâce Multi Collagen (Tip 1, 2, 3, 5)",
        "Booster Shot",
        "Booster Patch",
        "Immunplus",
        "Nutrimeal Live Fit",
        "Melatonin Sleep Patch",
        "Multicollagen 10.000 mg  Şase",
        "POWER & PAIN PATCH",
        "Gusto Classico Coffee Reishi mantar ekstraktı",
        "Anti Aging Longevity Serum (Hyaluronik Asit, Vitamin C, E, Koenzim Q10, Biotin)",
        "Flawless Skin Brightener (C Vitamini, Arbutin, Niasinamid)",
        "Youth Restoring Cleanser",
        "Intense Moisture Sunscreen (SPF 50, Hyaluronik Asit, Shea Butter)",
        "Regenerating Hair Shampoo (Papatya, Kamelya özleri)",
        "Whitening Fresh Toothpaste",
    ],

    # =========================
    # Siberian Wellness
    # =========================
   
    "siberian": [
        # Daily Box / kombine setler
        "Siberian Wellness BOOST BOX",
        "Siberian Wellness GLUCO Box",

        # Essential Botanics – bitkisel takviyeler
        "Siberian Wellness Essential Botanics. Aronia & Lutein",
        "Siberian Wellness Essential Botanics. Bearberry & Lingonberry",

        # Essential Minerals / Elemvitals – mineral destekler
        "Essential Minerals IODINE",
        "Essential Minerals IRON",
        "Essential Minerals MAGNESIUM",
        "Essential Minerals ORGANIC ZINC",
        "Essential Minerals ORGANIK KALSIYUM",
        "Essential Minerals SELENIUM",

        # Essential Sorbents / Lymphosan
        "Siberian Wellness Essential Sorbents JOINT COMFORT",
        "Siberian Wellness INULIN CONCENTRATE",

        # Essential Vitamins – vitamin destekleri
        "Alfa lipoik asit",
        "C vitamini ve rutin",
        "Siberian Wellness Essential Vitamins B-COMPLEX & BETAINE",
        "Siberian Wellness Essential Vitamins BEAUTY VITALS",
        "Siberian Wellness Essential Vitamins VITAMIN D3",

        # Novomin – multivitamin seti
        "Siberian Wellness N.V.M.N. FORMULA 4",

        # Genel wellness seti
        "Siberian Wellness RENAISSANCE TRIPLE SET",

        # SynchroVitals – kronobiyoloji destekleri
        "Siberian Health SYNCHROVITALS II",
        "Siberian Health SYNCHROVITALS IV",
        "Siberian Wellness SYNCHROVITALS V",

        # Trimegavitals – karotenoid ve göz destekleri
        "Beta-Carotene in Sea-Buckthorn Oil",
        "Siberian Wellness Trimegavitals, LUTEIN AND ZEAKSANTIN SUPERCONCANTRATE",

        # Vitamama – çocuk ve bağışıklık
        "Dino Vitamino Syrup with Vitamins and Minerals",
        "VITAMAMA Immunotops Syrup",

        # Women's Health serisi – kadın sağlığı
        "D-mannose & Cranberry extract",
        "Hyaluronic Acid & Natural Vitamin C",
        "Хронолонг",

        # Beslenme / genel güzellik desteği
        "Young & Beauty",
    ],

    # --------------------------------------------------------
    # Welltures Global
    # --------------------------------------------------------
    "welltures_global": [
        "welltures Gluwell",
        "welltures Omiwell",
        "welltures Multiwell",
        "welltures Suprawell",
        "welltures Epiwell",
        "welltures Bodywell",
        "welltures Admiwell",
        "welltures Maxiwell",
        "welltures Migwell",
        "welltures Frekanswell",
        "welltures Miraclewell",
        "welltures Collagen Face",
        "welltures Collagen Eye",
        "welltures Vitamin C Serum",
        "welltures Hyaluronic Acid Serum",
        "welltures Collagen Serum",
        "welltures Anti Aging Serum",
        "welltures Yüz Temizleme Köpüğü",
        "welltures Sun Stick",
        "welltures Cafewell",
    ],

        # --------------------------------------------------------
    # Herbalife
    # --------------------------------------------------------
    "herbalife": [
        "Formül 1 Shake",
        "Herbal Aloe Konsantre İçecek",
        "Formül 1 Çorba",
        "SKIN Collagen Drink Powder",
        "Bitkisel Çay Tozu",
        "Heartwell",
        "Tri Blend Select",
        "Protein Bar",
        "LiftOff",
        "Niteworks",
        "Protein Cips",
        "Formül 3 Pro-Boost",
        "Multi-fiber",
        "Herbalifeline Max",
        "Xtra-Cal",
        "Pro-Drink",
        "Vitamin-Mineral Kadınlar İçin",
        "Vitamin-Mineral Erkekler İçin",
        "Soğuk Kahve Karışımı",
        "Pro-core",
        "Thermo Complete",
        "Herbalife24 CR7 Drive",
        "Herbalife24 RB ProMax",
    ],

    # --------------------------------------------------------
    # Pati & Generic (Markasız)
    # --------------------------------------------------------
    "pati": [],

    # Markasız / genel analizde kullanılacak nötr ürün listesi
    "generic": [
        "multivitamin",
        "omega-3",
        "probiyotik",
        "koenzim q10",
        "kolajen takviyesi",
        "antioksidan destek karışımı",
        "detoks destek formülü",
        "karaciğer destek kompleksi",
        "bağışıklık güçlendirici kompleks",
        "magnezyum + B6 takviyesi",
        "D3 + K2 vitamini",
        "C vitamini takviyesi",
        "lif (fiber) takviyesi",
        "metabolizma destek karışımı",
        "bitkisel adaptogen destek",
    ],
}

# Alias tanımları
PRODUCT_LISTS["algo_tts"] = PRODUCT_LISTS.get("algo", [])
PRODUCT_LISTS["welltures"] = PRODUCT_LISTS.get("welltures_global", [])


def guess_gender_from_name(name: str) -> str:
    n = (name or "").lower()
    if "kadın" in n or "women" in n or "ladies" in n or "female" in n or "she" in n:
        return "female"
    if "erkek" in n or "men" in n or "male" in n or "he" in n:
        return "male"
    if "(kadın)" in n:
        return "female"
    if "(erkek)" in n:
        return "male"
    return "unisex"


BRAND_PRODUCTS_META: Dict[str, List[Dict[str, str]]] = {}
for key, plist in PRODUCT_LISTS.items():
    meta = []
    for p in plist:
        meta.append({"name": p, "gender": guess_gender_from_name(p)})
    BRAND_PRODUCTS_META[key] = meta


def load_brand_products_map() -> dict:
    return BRAND_PRODUCTS_META


ALL_BRAND_PRODUCTS_LOWER = {
    p.lower()
    for plist in PRODUCT_LISTS.values()
    for p in plist
}

ONEMORE_PRODUCTS_LOWER = {p.lower() for p in PRODUCT_LISTS.get("onemore", [])}


def get_brand_products(brand: str) -> List[str]:
    return PRODUCT_LISTS.get(brand, [])

PRODUCT_LISTS["algo_tts"] = PRODUCT_LISTS.get("algo", [])
PRODUCT_LISTS["welltures"] = PRODUCT_LISTS.get("welltures_global", [])

# ===============================================
# MARKA İSİMLERİ (BRAND_LABELS)
# - Arayüzde görünen temiz isimler burada
# ===============================================
BRAND_LABELS: dict[str, str] = {
    "generic": "Genel Wellness Analizi (Markasız)",

    "onemore": "OneMore International",
    "xaura": "Xaura Global",
    "atomy": "Atomy",
    "doctorem": "Doctorem International",

    # Algo / Algophyco TTS
    "algo": "Algophyco TTS",
    "algo_tts": "Algophyco TTS",

    "herbalife": "Herbalife",
    "welltures": "Welltures Global",
    "welltures_global": "Welltures Global",

    # Yeni eklediklerimiz
    "now": "Now International",
    "forever": "Forever Living",
    "siberian": "Siberian Wellness",
    "amare": "Amare Global",
    "onyedi": "Onyedi Wellness",
    "oxo": "Oxo Global",
    "som": "Som International",
    "indeva": "İndeva Global",

    # PatiPro
    "pati": "PatiPro Wellness",
}

# ====================================================
#  UI DİL ÇEVİRİLERİ
# ====================================================

UI_TRANSLATIONS = {
    "tr": {
        "report_title": "Kişisel Wellness Raporu",
        "complaint_report_title": "Şikâyet Bazlı Wellness Raporu",
        "detail_label_ultra": "Ultra Detaylı (En kapsamlı)",
        "nav_kisi": "Kişisel Bilgiler",
        "nav_genel": "Genel Bulgu & Yaşam Stratejisi",
        "nav_plan": "4 Haftalık Yaşam Planı",
        "nav_risk10": "En Riskli 10 Sistem",
        "nav_kanallar": "Kanallar & Kollateraller",
        "nav_bilinc": "İnsan Bilinç Düzeyi",
        "nav_kartlar": "46 Sistem Kartı",
        "nav_uyari": "Tıbbi Sorumluluk Beyanı",
        "subtitle_46": "Her bir kart ilgili sistem başlığını gösterir; tüm sistemler listelenir.",
        "no_detail_card": "Bu kart için detaylı analiz bulunmamaktadır.",
        "no_product_suggestion": "Bu kart için spesifik ürün önerisi yoktur.",
        "btn_back": "Ana Sayfaya Dön",
        "btn_new": "Yeni Test",
        "btn_pdf": "Analizi PDF Olarak Kaydet",
        "vucut_title": "Vücut Formu Değerlendirmesi",
    },
    "it": {
        "report_title": "Rapporto di benessere personale",
        "detail_label_ultra": "Analisi ultra dettagliata (la più completa)",
        "nav_kisi": "Dati personali",
        "nav_genel": "Risultati generali & strategia di stile di vita",
        "nav_plan": "Piano di vita di 4 settimane",
        "nav_risk10": "I 10 sistemi più vulnerabili",
        "nav_kanallar": "Canali & collaterali",
        "nav_bilinc": "Livello di coscienza umana",
        "nav_kartlar": "46 schede di sistema",
        "nav_uyari": "Avvertenza medica",
        "subtitle_46": "Il titolo di ogni scheda mostra il sistema corrispondente; tutti i sistemi sono elencati.",
        "no_detail_card": "Nessuna analisi dettagliata disponibile per questa scheda.",
        "no_product_suggestion": "Nessun suggerimento specifico di prodotto per questa scheda.",
        "btn_back": "Torna alla Home",
        "btn_new": "Nuovo test",
        "btn_pdf": "Salva l'analisi in PDF",
        "vucut_title": "Valutazione della forma fisica",
    },
    "az": {
        "report_title": "Şəxsi Wellness Hesabatı",
        "complaint_report_title": "Şikayət Əsaslı Wellness Hesabatı",
        "detail_label_ultra": "Çox detallı analiz (ən əhatəlisi)",
        "nav_kisi": "Şəxsi məlumatlar",
        "nav_genel": "Ümumi nəticələr və həyat tərzi strategiyası",
        "nav_plan": "4 həftəlik həyat planı",
        "nav_risk10": "Ən riskli 10 sistem",
        "nav_kanallar": "Kanallar və kollaterallar",
        "nav_bilinc": "İnsan şüur səviyyəsi",
        "nav_kartlar": "46 sistem kartı",
        "nav_uyari": "Tibbi məsuliyyət bəyanatı",
        "subtitle_46": "Hər kart müvafiq sistem başlığını göstərir; bütün sistemlər siyahılanıb.",
        "no_detail_card": "Bu kart üçün ətraflı analiz yoxdur.",
        "no_product_suggestion": "Bu kart üçün xüsusi məhsul tövsiyəsi yoxdur.",
        "btn_back": "Ana səhifəyə qayıt",
        "btn_new": "Yeni test",
        "btn_pdf": "Analizi PDF kimi saxla",
        "vucut_title": "Bədən formunun qiymətləndirilməsi",
    },    
    "en": {
        "report_title": "Personal Wellness Report",
        "complaint_report_title": "Complaint-Based Wellness Report",
        "detail_label_ultra": "Ultra Detailed (Most comprehensive)",
        "nav_kisi": "Personal Information",
        "nav_genel": "General Findings & Lifestyle Strategy",
        "nav_plan": "4-Week Lifestyle Plan",
        "nav_risk10": "Top 10 Most Vulnerable Systems",
        "nav_kanallar": "Meridians & Collaterals",
        "nav_bilinc": "Human Consciousness Level",
        "nav_kartlar": "46 System Cards",
        "nav_uyari": "Medical Disclaimer",
        "subtitle_46": "Each card title shows the corresponding system; all systems are listed.",
        "no_detail_card": "No detailed analysis is available for this card.",
        "no_product_suggestion": "No specific product suggestion is available for this card.",
        "btn_back": "Back to Home",
        "btn_new": "New Analysis",
        "btn_pdf": "Save Analysis as PDF",
        "vucut_title": "Body Composition",
    },
    "de": {
        "report_title": "Persönlicher Wellness-Bericht",
        "complaint_report_title": "Beschwerdebasierter Wellnessbericht",
        "detail_label_ultra": "Ultradetaillierte Analyse (am umfassendsten)",
        "nav_kisi": "Persönliche Daten",
        "nav_genel": "Allgemeine Befunde & Lebensstrategie",
        "nav_plan": "4-Wochen-Lebensplan",
        "nav_risk10": "Die 10 risikoreichsten Systeme",
        "nav_kanallar": "Kanäle & Kollaterale",
        "nav_bilinc": "Bewusstseinsniveau des Menschen",
        "nav_kartlar": "46 Systemkarten",
        "nav_uyari": "Medizinischer Haftungsausschluss",
        "subtitle_46": "Der Titel jeder Karte enthält den Systemnamen; alle Systeme werden aufgelistet.",
        "no_detail_card": "Für diese Karte liegt keine detaillierte Analyse vor.",
        "no_product_suggestion": "Für diese Karte gibt es keine spezifische Produktempfehlung.",
        "btn_back": "Zur Startseite",
        "btn_new": "Neue Analyse",
        "btn_pdf": "Analyse als PDF speichern",
        "vucut_title": "Körperzusammensetzung",
    },
    "es": {
        "report_title": "Informe de bienestar personal",
        "complaint_report_title": "Informe de bienestar basado en quejas",
        "detail_label_ultra": "Análisis ultra detallado (el más completo)",
        "nav_kisi": "Información personal",
        "nav_genel": "Hallazgos generales y estrategia de estilo de vida",
        "nav_plan": "Plan de vida de 4 semanas",
        "nav_risk10": "Los 10 sistemas más vulnerables",
        "nav_kanallar": "Canales y colaterales",
        "nav_bilinc": "Nivel de conciencia humana",
        "nav_kartlar": "Tarjetas de los 46 sistemas",
        "nav_uyari": "Descargo de responsabilidad médica",
        "subtitle_46": "El título de cada tarjeta muestra el sistema correspondiente; todos los sistemas están listados.",
        "no_detail_card": "No hay un análisis detallado disponible para esta tarjeta.",
        "no_product_suggestion": "No hay recomendaciones específicas de productos para esta tarjeta.",
        "btn_back": "Volver al inicio",
        "btn_new": "Nuevo análisis",
        "btn_pdf": "Guardar análisis en PDF",
        "vucut_title": "Composición corporal",
    },
    "pt": {
        "report_title": "Relatório de bem-estar pessoal",
        "complaint_report_title": "Relatório de bem-estar baseado em queixas",
        "detail_label_ultra": "Análise ultra detalhada (a mais completa)",
        "nav_kisi": "Informações pessoais",
        "nav_genel": "Achados gerais e estratégia de estilo de vida",
        "nav_plan": "Plano de vida de 4 semanas",
        "nav_risk10": "Top 10 sistemas mais vulneráveis",
        "nav_kanallar": "Canais e colaterais",
        "nav_bilinc": "Nível de consciência humana",
        "nav_kartlar": "Cartas dos 46 sistemas",
        "nav_uyari": "Aviso de responsabilidade médica",
        "subtitle_46": "O título de cada carta mostra o sistema correspondente; todos os sistemas estão listados.",
        "no_detail_card": "Não há análise detalhada disponível para esta carta.",
        "no_product_suggestion": "Não há recomendação específica de produto para esta carta.",
        "btn_back": "Voltar à página inicial",
        "btn_new": "Nova análise",
        "btn_pdf": "Salvar análise em PDF",
        "vucut_title": "Composição corporal",
    },
    "fr": {
        "report_title": "Rapport de bien-être personnel",
        "complaint_report_title": "Rapport de bien-être basé sur les plaintes",
        "detail_label_ultra": "Analyse ultra détaillée (la plus complète)",
        "nav_kisi": "Informations personnelles",
        "nav_genel": "Résultats généraux & stratégie de vie",
        "nav_plan": "Plan de vie sur 4 semaines",
        "nav_risk10": "Les 10 systèmes les plus vulnérables",
        "nav_kanallar": "Canaux & collatéraux",
        "nav_bilinc": "Niveau de conscience humaine",
        "nav_kartlar": "Cartes des 46 systèmes",
        "nav_uyari": "Clause de non-responsabilité médicale",
        "subtitle_46": "Le titre de chaque carte indique le système concerné ; tous les systèmes sont listés.",
        "no_detail_card": "Aucune analyse détaillée n’est disponibile pour cette carte.",
        "no_product_suggestion": "Aucune recommandation de produit spécifique pour cette carte.",
        "btn_back": "Retour à l’accueil",
        "btn_new": "Nouvelle analyse",
        "btn_pdf": "Enregistrer l’analyse en PDF",
        "vucut_title": "Composition corporelle",
    },
    "ru": {
        "report_title": "Персональный отчёт о состоянии здоровья",
        "complaint_report_title": "Отчет о здоровье на основе жалоб",
        "detail_label_ultra": "Ультрадетальный анализ (самый полный)",
        "nav_kisi": "Личная информация",
        "nav_genel": "Общие выводы и стратегия образа жизни",
        "nav_plan": "4-недельный план образа жизни",
        "nav_risk10": "10 наиболее уязвимых систем",
        "nav_kanallar": "Каналы и коллатералы",
        "nav_bilinc": "Уровень человеческого сознания",
        "nav_kartlar": "46 карточек систем",
        "nav_uyari": "Медицинский отказ от ответственности",
        "subtitle_46": "В названии каждой карточки указан соответствующий орган/система; перечислены все системы.",
        "no_detail_card": "Для этой карточки подробный анализ отсутствует.",
        "no_product_suggestion": "Для этой карточки нет конкретных рекомендаций по продуктам.",
        "btn_back": "Вернуться на главную",
        "btn_new": "Новый анализ",
        "btn_pdf": "Сохранить анализ в формате PDF",
        "vucut_title": "Состав тела",
    },
    "ar": {
        "report_title": "تقرير العافية الشخصي",
        "complaint_report_title": "تقرير العافية القائم على الشكاوى",
        "detail_label_ultra": "تحليل مفصل للغاية (الأكثر شمولاً)",
        "nav_kisi": "المعلومات الشخصية",
        "nav_genel": "النتائج العامة واستراتيجية نمط الحياة",
        "nav_plan": "خطة نمط الحياة لمدة 4 أسابيع",
        "nav_risk10": "أكثر 10 أنظمة عرضة للخطر",
        "nav_kanallar": "القنوات والضمانات",
        "nav_bilinc": "مستوى الوعي البشري",
        "nav_kartlar": "بطاقات الأنظمة الـ 46",
        "nav_uyari": "إخلاء المسؤولية الطبية",
        "subtitle_46": "يحتوي عنوان كل بطاقة على اسم النظام؛ جميع الأنظمة مُدرجة.",
        "no_detail_card": "لا توجد تحليلات مفصلة لهذه البطاقة.",
        "no_product_suggestion": "لا توجد توصيات محددة للمنتجات لهذه البطاقة.",
        "btn_back": "العودة إلى الصفحة الرئيسية",
        "btn_new": "تحليل جديد",
        "btn_pdf": "حفظ التحليل كملف PDF",
        "vucut_title": "تركيب الجسم",
    },
    "fa": {
        "report_title": "گزارش سلامتی فردی",
        "complaint_report_title": "گزارش تندرستی بر پایه شکایات",
        "detail_label_ultra": "تحلیل فوق‌العاده دقیق (کامل‌ترین)",
        "nav_kisi": "اطلاعات شخصی",
        "nav_genel": "یافته‌های کلی و استراتژی سبک زندگی",
        "nav_plan": "برنامه ۴ هفته‌ای زندگی",
        "nav_risk10": "۱۰ سیستم با بیشترین ریسک",
        "nav_kanallar": "کانال‌ها و مسیرها",
        "nav_bilinc": "سطح آگاهی انسان",
        "nav_kartlar": "۴۶ کارت سیستم",
        "nav_uyari": "سلب مسئولیت پزشکی",
        "subtitle_46": "نام هر سیستم در عنوان کارت نوشته شده است؛ تمام سیستم‌ها فهرست شده‌اند.",
        "no_detail_card": "برای این کارت تحلیل مفصلی ثبت نشده است.",
        "no_product_suggestion": "برای این کارت توصیه خاصی برای محصول وجود ندارد.",
        "btn_back": "بازگشت به صفحه اصلی",
        "btn_new": "تحلیل جدید",
        "btn_pdf": "ذخیره تحلیل به صورت PDF",
        "vucut_title": "ترکیب بدن",
    },
    "mn": {
        "report_title": "Хувийн сайн сайхны тайлан",
        "complaint_report_title": "Гомдолд суурилсан сайн сайхны тайлан",
        "detail_label_ultra": "Маш дэлгэрэнгүй шинжилгээ (хамгийн бүрэн)",
        "nav_kisi": "Хувийн мэдээлэл",
        "nav_genel": "Ерөнхий дүгнэлт ба амьдралын хэв маягийн стратеги",
        "nav_plan": "4 долоо хоногийн амьдралын төлөвлөгөө",
        "nav_risk10": "Эрсдэл өндөртэй 10 систем",
        "nav_kanallar": "Сувгууд ба коллатералууд",
        "nav_bilinc": "Хүний ухамсрын түвшин",
        "nav_kartlar": "46 системийн карт",
        "nav_uyari": "Эмнэлгийн хариуцлагаас татгалзсан мэдэгдэл",
        "subtitle_46": "Карта бүрийн нэр дээр тухайн систем бичигдсэн; бүх систем жагсаагдсан.",
        "no_detail_card": "Энэ картын дэлгэрэнгүй шинжилгээ байхгүй.",
        "no_product_suggestion": "Энэ картанд тусгай бүтээгдэхүүн санал болгоогүй.",
        "btn_back": "Нүүр хуудас руу буцах",
        "btn_new": "Шинэ шинжилгээ",
        "btn_pdf": "Тайланг PDF хэлбэрээр хадгалах",
        "vucut_title": "Биеийн бүтцийн үзүүлэлт",
    },
    "id": {
        "report_title": "Laporan kebugaran pribadi",
        "complaint_report_title": "Laporan Kesehatan Berbasis Keluhan",
        "detail_label_ultra": "Analisis sangat detail (paling lengkap)",
        "nav_kisi": "Informasi pribadi",
        "nav_genel": "Temuan umum & strategi gaya hidup",
        "nav_plan": "Rencana hidup 4 minggu",
        "nav_risk10": "10 sistem paling rentan",
        "nav_kanallar": "Kanal & kolateral",
        "nav_bilinc": "Tingkat kesadaran manusia",
        "nav_kartlar": "46 kartu sistem",
        "nav_uyari": "Penafian medis",
        "subtitle_46": "Judul setiap kartu menunjukkan sistem terkait; semua sistem tercantum.",
        "no_detail_card": "Tidak ada analisis detail untuk kartu ini.",
        "no_product_suggestion": "Tidak ada rekomendasi produk khusus untuk kartu ini.",
        "btn_back": "Kembali ke beranda",
        "btn_new": "Analisis baru",
        "btn_pdf": "Simpan analisis sebagai PDF",
        "vucut_title": "Komposisi tubuh",
    },
}

LANGUAGE_LABELS = {
    "auto": "Otomatik (PDF/metin dilini algıla)",
    "tr": "Türkçe",
    "en": "İngilizce",
    "de": "Almanca",
    "es": "İspanyolca",
    "pt": "Portekizce",
    "fr": "Fransızca",
    "ru": "Rusça",
    "ar": "Arapça",
    "fa": "Farsça",
    "mn": "Moğolca",
    "id": "Endonezce",
    "it": "İtalyanca",
    "az": "Azerbaycanca",
}

# ====================================================
# RAPOR İÇİ ETİKETLER (11 DİL) + 2 DİLLİ METİN ÜRETİCİ
# ====================================================

BASE_REPORT_LABELS = {
    "tr": {
        "report_title": "Kişisel Wellness Raporu",
        "section_personal_data": "Kişisel Bilgiler",
        "label_name": "Ad Soyad",
        "label_age": "Yaş",
        "label_gender": "Cinsiyet",
        "label_height": "Boy",
        "label_weight": "Kilo",
        "label_test_date": "Test Tarihi",

        "section_body_form": "Vücut Formu Değerlendirmesi",
        "label_general_tag": "Genel Etiket",
        "label_form_score": "Form Puanı",
        "label_ratio": "Oran",
        "label_bmr": "BMR",

        "label_status": "Durum",
        "label_symptoms": "Belirtiler",
        "label_risks": "Riskler",
        "label_lifestyle": "Yaşam Tavsiyesi",
        "label_lifestyle_alt": "Yaşam Önerileri",
        "label_detailed_explanation": "Detaylı Açıklama",
        "label_product_recommendations": "Ürün Önerileri",
        "label_recommended_products": "Önerilen Ürünler",
        "label_product_usage": "Ürün Kullanımı",

        "section_general_products": "Genel Ürün Önerileri",
        "section_46_cards": "46 sistem kartı",
        "note_46_cards": "46 kartın her biri ayrı bir sistem başlığına karşılık gelir.",

        "section_channels": "Kanallar ve kollateraller",
        "section_consciousness": "İnsan Bilinç Düzeyi",
        "section_4week_plan": "4 Haftalık Yaşam Planı",
        "section_medical_disclaimer": "Tıbbi Sorumluluk Beyanı",
    },

    "en": {
        "report_title": "Personal Wellness Report",
        "section_personal_data": "Personal Data",
        "label_name": "Full Name",
        "label_age": "Age",
        "label_gender": "Gender",
        "label_height": "Height",
        "label_weight": "Weight",
        "label_test_date": "Test Date",

        "section_body_form": "Body Composition Assessment",
        "label_general_tag": "Overall Tag",
        "label_form_score": "Form Score",
        "label_ratio": "Ratio",
        "label_bmr": "BMR",

        "label_status": "Status",
        "label_symptoms": "Symptoms",
        "label_risks": "Risks",
        "label_lifestyle": "Lifestyle Advice",
        "label_lifestyle_alt": "Lifestyle Suggestions",
        "label_detailed_explanation": "Detailed Explanation",
        "label_product_recommendations": "Product Recommendations",
        "label_recommended_products": "Recommended Products",
        "label_product_usage": "Product Usage",

        "section_general_products": "General Product Recommendations",
        "section_46_cards": "46 System Cards",
        "note_46_cards": "Each of the 46 cards corresponds to a specific body system.",

        "section_channels": "Channels and Collaterals",
        "section_consciousness": "Human Consciousness Level",
        "section_4week_plan": "4-Week Lifestyle Plan",
        "section_medical_disclaimer": "Medical Responsibility Statement",
    },

    "it": {
        "report_title": "Rapporto di benessere personale",
        "section_personal_data": "Dati personali",
        "label_name": "Nome e cognome",
        "label_age": "Età",
        "label_gender": "Sesso",
        "label_height": "Altezza",
        "label_weight": "Peso",
        "label_test_date": "Data del test",

        "section_body_form": "Valutazione della forma fisica",
        "label_general_tag": "Etichetta generale",
        "label_form_score": "Punteggio della forma fisica",
        "label_ratio": "Percentuale",
        "label_bmr": "BMR",

        "label_status": "Stato",
        "label_symptoms": "Sintomi",
        "label_risks": "Rischi",
        "label_lifestyle": "Raccomandazioni sullo stile di vita",
        "label_lifestyle_alt": "Suggerimenti sullo stile di vita",
        "label_detailed_explanation": "Spiegazione dettagliata",
        "label_product_recommendations": "Raccomandazioni sui prodotti",
        "label_recommended_products": "Prodotti consigliati",
        "label_product_usage": "Uso del prodotto",

        "section_general_products": "Raccomandazioni generali sui prodotti",
        "section_46_cards": "46 schede di sistema",
        "note_46_cards": "Ogni scheda corrisponde a un sistema specifico dell'organismo.",

        "section_channels": "Canali & collaterali",
        "section_consciousness": "Livello di coscienza umana",
        "section_4week_plan": "Piano di vita di 4 settimane",
        "section_medical_disclaimer": "Dichiarazione di responsabilità medica",
    },

    "ar": {
        "report_title": "تقرير العافية الشخصي",
        "section_personal_data": "المعلومات الشخصية",
        "label_name": "الاسم والكنية",
        "label_age": "العمر",
        "label_gender": "الجنس",
        "label_height": "الطول",
        "label_weight": "الوزن",
        "label_test_date": "تاريخ الفحص",

        "section_body_form": "تقييم تركيبة الجسم",
        "label_general_tag": "التوصيف العام",
        "label_form_score": "درجة تقييم البنية",
        "label_ratio": "النسبة المئوية",
        "label_bmr": "معدل الاستقلاب الأساسي (BMR)",

        "label_status": "الحالة",
        "label_symptoms": "الأعراض",
        "label_risks": "المخاطر",
        "label_lifestyle": "نصائح لنمط الحياة",
        "label_lifestyle_alt": "اقتراحات لنمط الحياة",
        "label_detailed_explanation": "شرح تفصيلي",
        "label_product_recommendations": "توصيات بالمنتجات",
        "label_recommended_products": "المنتجات الموصى بها",
        "label_product_usage": "طريقة استخدام المنتج",

        "section_general_products": "التوصيات العامة للمنتجات",
        "section_46_cards": "بطاقات الأنظمة الـ 46",
        "note_46_cards": "كل بطاقة من البطاقات الـ 46 تمثل نظامًا محددًا في الجسم.",

        "section_channels": "القنوات والمسارات الجانبية",
        "section_consciousness": "مستوى الوعي البشري",
        "section_4week_plan": "خطة الحياة لمدة 4 أسابيع",
        "section_medical_disclaimer": "بيان المسؤولية الطبية",
    },

    # Diğer diller (de, es, fr, pt, ru, fa, id, mn ...) için
    # etiket girmezsen, fonksiyon İngilizce veya Türkçe'ye düşer.
}

# ====================================================
#  ARAYÜZ METİNLERİ (SOL MENÜ, BUTONLAR) - 1. ve 2. DİL
# ====================================================

def build_ui_texts(target_lang: str, second_lang: str) -> dict:
    """
    Arayüz metinlerini hazırlar.

    KURAL:
    - Her zaman 1. dil (target_lang) esastır.
    - 2. dil seçilmişse, aynı metni "1.dil / 2.dil" şeklinde tek satırda birleştiririz.
      Böylece sol menü ve butonlar iki dili de aynı anda gösterir.
    - 2. dil boşsa, sadece 1. dilde metin döner.
    """
    primary = (target_lang or "tr").lower()
    if primary not in UI_TRANSLATIONS:
        primary = "tr"

    # 2. dil geçersizse veya yoksa tek dilli çalış
    second = (second_lang or "").lower()
    if second not in UI_TRANSLATIONS:
        second = ""

    base_primary = UI_TRANSLATIONS[primary]

    # Tek dilli kullanım
    if not second:
        return base_primary

    base_second = UI_TRANSLATIONS[second]

    combined = {}
    all_keys = set(base_primary.keys()) | set(base_second.keys())
    for key in all_keys:
        p = base_primary.get(key, "")
        s = base_second.get(key, "")
        if p and s:
            combined[key] = f"{p} / {s}"
        else:
            # birinde yoksa olanı kullan
            combined[key] = p or s
    return combined


# ====================================================
#  RAPOR İÇİ BAŞLIK & ETİKET METİNLERİ - 1. ve 2. DİL
# ====================================================

LABEL_TEXTS: Dict[str, Dict[str, str]] = {
    # ---------------- TR ----------------
    "tr": {
        "report_title": "Kişisel Wellness Raporu",
        "complaint_report_title": "Şikâyet Bazlı Wellness Raporu",
        "section_personal_data": "Kişisel Bilgiler",
        "section_body_form": "Vücut Formu Değerlendirmesi",
        "section_general_products": "Genel Ürün Önerileri",
        "section_46_cards": "46 Sistem Kartı",
        "section_medical_disclaimer": "Tıbbi Sorumluluk Beyanı",

        "label_sections": "Bölümler",
        "label_detail_level": "Detay seviyesi",

        "label_name": "Ad Soyad",
        "label_age": "Yaş",
        "label_gender": "Cinsiyet",
        "label_height": "Boy",
        "label_weight": "Kilo",
        "label_test_date": "Test Tarihi",

        "label_general_tag": "Genel Etiket",
        "label_form_score": "Form Puanı",
        "label_ratio": "Oran",
        "label_bmr": "Bazal Metabolizma Hızı (BMR)",

        "label_status": "Durum",
        "label_symptoms": "Belirtiler",
        "label_risks": "Riskler",
        "label_lifestyle": "Yaşam Önerileri",
        "label_product_recommendations": "Ürün Önerileri",
        "label_product_usage": "Ürün Kullanımı",
        "label_detailed_explanation": "Detaylı Açıklama",

        "badge_46_cards_hint": "46 kartın her biri ayrı bir sistem başlığına karşılık gelir.",

        "label_no_risk_list": "Bu raporda vurgulanan riskli sistem listesi üretilemedi.",
        "label_no_plan": "Bu rapor için 4 haftalık plan üretilmedi.",
        "label_no_general_products": "Genel düzeyde spesifik ürün önerisi yok.",
        "label_no_system_cards": "Bu rapor için sistem bazlı kartlar üretilemedi.",
    },

    # ---------------- EN ----------------
    "en": {
        "report_title": "Personal Wellness Report",
        "complaint_report_title": "Complaint-Based Wellness Report",
        "section_personal_data": "Personal Information",
        "section_body_form": "Body Composition Evaluation",
        "section_general_products": "General Product Recommendations",
        "section_46_cards": "46 System Cards",
        "section_medical_disclaimer": "Medical Responsibility Statement",

        "label_sections": "Sections",
        "label_detail_level": "Detail level",

        "label_name": "Name & Surname",
        "label_age": "Age",
        "label_gender": "Gender",
        "label_height": "Height",
        "label_weight": "Weight",
        "label_test_date": "Test Date",

        "label_general_tag": "General Tag",
        "label_form_score": "Form Score",
        "label_ratio": "Ratio",
        "label_bmr": "Basal Metabolic Rate (BMR)",

        "label_status": "Status",
        "label_symptoms": "Symptoms",
        "label_risks": "Risks",
        "label_lifestyle": "Lifestyle Recommendations",
        "label_product_recommendations": "Product Recommendations",
        "label_product_usage": "Product Usage",
        "label_detailed_explanation": "Detailed Explanation",

        "badge_46_cards_hint": "Each of the 46 cards corresponds to a different body system.",

        "label_no_risk_list": "No highlighted high-risk system list was generated for this report.",
        "label_no_plan": "No 4-week plan was generated for this report.",
        "label_no_general_products": "No general-level specific product recommendation is available.",
        "label_no_system_cards": "No system-based cards were generated for this report.",
    },

    # ---------------- IT ----------------
    "it": {
        "report_title": "Rapporto di benessere personale",
        "complaint_report_title": "Rapporto di benessere basato sui sintomi",
        "section_personal_data": "Dati personali",
        "section_body_form": "Valutazione della forma fisica",
        "section_general_products": "Raccomandazioni generali di prodotti",
        "section_46_cards": "46 schede di sistema",
        "section_medical_disclaimer": "Dichiarazione di responsabilità medica",

        "label_sections": "Sezioni",
        "label_detail_level": "Livello di dettaglio",

        "label_name": "Nome e cognome",
        "label_age": "Età",
        "label_gender": "Sesso",
        "label_height": "Altezza",
        "label_weight": "Peso",
        "label_test_date": "Data del test",

        "label_general_tag": "Etichetta generale",
        "label_form_score": "Punteggio forma",
        "label_ratio": "Rapporto",
        "label_bmr": "Metabolismo basale (BMR)",

        "label_status": "Stato",
        "label_symptoms": "Sintomi",
        "label_risks": "Rischi",
        "label_lifestyle": "Raccomandazioni sullo stile di vita",
        "label_product_recommendations": "Raccomandazioni di prodotti",
        "label_product_usage": "Uso dei prodotti",
        "label_detailed_explanation": "Spiegazione dettagliata",

        "badge_46_cards_hint": "Ognuna delle 46 schede corrisponde a un diverso sistema del corpo.",

        "label_no_risk_list": "Nessun elenco di sistemi ad alto rischio è stato generato per questo rapporto.",
        "label_no_plan": "Nessun piano di 4 settimane è stato generato per questo rapporto.",
        "label_no_general_products": "Nessuna raccomandazione di prodotto generale è disponibile.",
        "label_no_system_cards": "Nessuna scheda di sistema è stata generata per questo rapporto.",
    },

    "az": {
        "report_title": "Şəxsi Wellness Hesabatı",
        "complaint_report_title": "Şikayət əsaslı Wellness Hesabatı",

        "section_personal_data": "Şəxsi Məlumatlar",
        "section_body_form": "Bədən Formasının Qiymətləndirilməsi",
        "section_general_products": "Ümumi Məhsul Tövsiyələri",
        "section_46_cards": "46 Sistem Kartı",
        "section_medical_disclaimer": "Tibbi Məsuliyyət Bəyanatı",

        "label_sections": "Bölmələr",
        "label_detail_level": "Detallılıq səviyyəsi",

        "label_name": "Ad və Soyad",
        "label_age": "Yaş",
        "label_gender": "Cinsiyyət",
        "label_height": "Boy",
        "label_weight": "Çəki",
        "label_test_date": "Test Tarixi",

        "label_general_tag": "Ümumi Etiket",
        "label_form_score": "Form Balı",
        "label_ratio": "Nisbət",
        "label_bmr": "Bazal Metabolizma Sürəti (BMR)",

        "label_status": "Vəziyyət",
        "label_symptoms": "Əlamətlər",
        "label_risks": "Risklər",
        "label_lifestyle": "Həyat tərzi tövsiyələri",
        "label_product_recommendations": "Məhsul tövsiyələri",
        "label_product_usage": "Məhsul istifadəsi",
        "label_detailed_explanation": "Ətraflı izah",

        "badge_46_cards_hint": "46 kartın hər biri ayrıca bir sistem başlığına uyğun gəlir.",

        "label_no_risk_list": "Bu hesabat üçün vurğulanmış riskli sistem siyahısı yaradılmadı.",
        "label_no_plan": "Bu hesabat üçün 4 həftəlik plan yaradılmadı.",
        "label_no_general_products": "Ümumi səviyyədə xüsusi məhsul tövsiyəsi yoxdur.",
        "label_no_system_cards": "Bu hesabat üçün sistem əsaslı kartlar yaradılmadı.",
    },

    # ---------------- ES ----------------
    "es": {
        "report_title": "Informe de bienestar personal",
        "complaint_report_title": "Informe de bienestar basado en quejas",
        "section_personal_data": "Datos personales",
        "section_body_form": "Evaluación de la composición corporal",
        "section_general_products": "Recomendaciones generales de productos",
        "section_46_cards": "46 tarjetas de sistema",
        "section_medical_disclaimer": "Declaración de responsabilidad médica",

        "label_sections": "Secciones",
        "label_detail_level": "Nivel de detalle",

        "label_name": "Nombre y apellidos",
        "label_age": "Edad",
        "label_gender": "Género",
        "label_height": "Altura",
        "label_weight": "Peso",
        "label_test_date": "Fecha del test",

        "label_general_tag": "Etiqueta general",
        "label_form_score": "Puntuación de forma",
        "label_ratio": "Relación",
        "label_bmr": "Tasa metabólica basal (BMR)",

        "label_status": "Estado",
        "label_symptoms": "Síntomas",
        "label_risks": "Riesgos",
        "label_lifestyle": "Recomendaciones de estilo de vida",
        "label_product_recommendations": "Recomendaciones de productos",
        "label_product_usage": "Uso de productos",
        "label_detailed_explanation": "Explicación detallada",

        "badge_46_cards_hint": "Cada una de las 46 tarjetas corresponde a un sistema corporal diferente.",

        "label_no_risk_list": "No se generó una lista de sistemas de alto riesgo para este informe.",
        "label_no_plan": "No se generó un plan de 4 semanas para este informe.",
        "label_no_general_products": "No hay recomendaciones generales de productos disponibles.",
        "label_no_system_cards": "No se generaron tarjetas de sistema para este informe.",
    },

    # ---------------- PT ----------------
    "pt": {
        "report_title": "Relatório de bem-estar pessoal",
        "complaint_report_title": "Relatório de bem-estar baseado em queixas",
        "section_personal_data": "Dados pessoais",
        "section_body_form": "Avaliação da composição corporal",
        "section_general_products": "Recomendações gerais de produtos",
        "section_46_cards": "46 cartões de sistema",
        "section_medical_disclaimer": "Declaração de responsabilidade médica",

        "label_sections": "Seções",
        "label_detail_level": "Nível de detalhe",

        "label_name": "Nome e sobrenome",
        "label_age": "Idade",
        "label_gender": "Gênero",
        "label_height": "Altura",
        "label_weight": "Peso",
        "label_test_date": "Data do teste",

        "label_general_tag": "Etiqueta geral",
        "label_form_score": "Pontuação de forma",
        "label_ratio": "Proporção",
        "label_bmr": "Taxa metabólica basal (BMR)",

        "label_status": "Estado",
        "label_symptoms": "Sintomas",
        "label_risks": "Riscos",
        "label_lifestyle": "Recomendações de estilo de vida",
        "label_product_recommendations": "Recomendações de produtos",
        "label_product_usage": "Uso de produtos",
        "label_detailed_explanation": "Explicação detalhada",

        "badge_46_cards_hint": "Cada um dos 46 cartões corresponde a um sistema corporal diferente.",

        "label_no_risk_list": "Não foi gerada uma lista de sistemas de alto risco para este relatório.",
        "label_no_plan": "Não foi gerado um plano de 4 semanas para este relatório.",
        "label_no_general_products": "Não há recomendações gerais de produtos disponíveis.",
        "label_no_system_cards": "Não foram gerados cartões de sistema para este relatório.",
    },

    # ---------------- FR ----------------
    "fr": {
        "report_title": "Rapport de bien-être personnel",
        "complaint_report_title": "Rapport de bien-être basé sur les plaintes",
        "section_personal_data": "Informations personnelles",
        "section_body_form": "Évaluation de la composition corporelle",
        "section_general_products": "Recommandations générales de produits",
        "section_46_cards": "46 cartes de système",
        "section_medical_disclaimer": "Déclaration de responsabilité médicale",

        "label_sections": "Sections",
        "label_detail_level": "Niveau de détail",

        "label_name": "Nom et prénom",
        "label_age": "Âge",
        "label_gender": "Genre",
        "label_height": "Taille",
        "label_weight": "Poids",
        "label_test_date": "Date du test",

        "label_general_tag": "Étiquette générale",
        "label_form_score": "Score de forme",
        "label_ratio": "Ratio",
        "label_bmr": "Métabolisme de base (BMR)",

        "label_status": "État",
        "label_symptoms": "Symptômes",
        "label_risks": "Risques",
        "label_lifestyle": "Recommandations de mode de vie",
        "label_product_recommendations": "Recommandations de produits",
        "label_product_usage": "Utilisation des produits",
        "label_detailed_explanation": "Explication détaillée",

        "badge_46_cards_hint": "Chacune des 46 cartes correspond à un système corporel différent.",

        "label_no_risk_list": "Aucune liste de systèmes à haut risque n’a été générée pour ce rapport.",
        "label_no_plan": "Aucun plan de 4 semaines n’a été généré pour ce rapport.",
        "label_no_general_products": "Aucune recommandation générale de produits n’est disponible.",
        "label_no_system_cards": "Aucune carte de système n’a été générée pour ce rapport.",
    },

    # ---------------- DE ----------------
    "de": {
        "report_title": "Persönlicher Wellnessbericht",
        "complaint_report_title": "Beschwerdebasierter Wellness-Bericht",
        "section_personal_data": "Persönliche Daten",
        "section_body_form": "Bewertung der Körperzusammensetzung",
        "section_general_products": "Allgemeine Produktempfehlungen",
        "section_46_cards": "46 Systemkarten",
        "section_medical_disclaimer": "Medizinischer Haftungsausschluss",

        "label_sections": "Abschnitte",
        "label_detail_level": "Detailgrad",

        "label_name": "Name und Nachname",
        "label_age": "Alter",
        "label_gender": "Geschlecht",
        "label_height": "Größe",
        "label_weight": "Gewicht",
        "label_test_date": "Testdatum",

        "label_general_tag": "Allgemeines Etikett",
        "label_form_score": "Form-Score",
        "label_ratio": "Verhältnis",
        "label_bmr": "Grundumsatz (BMR)",

        "label_status": "Status",
        "label_symptoms": "Symptome",
        "label_risks": "Risiken",
        "label_lifestyle": "Lebensstil-Empfehlungen",
        "label_product_recommendations": "Produktempfehlungen",
        "label_product_usage": "Produktverwendung",
        "label_detailed_explanation": "Detaillierte Erklärung",

        "badge_46_cards_hint": "Jede der 46 Karten entspricht einem anderen Körpersystem.",

        "label_no_risk_list": "Für diesen Bericht wurde keine Liste der Hochrisikosysteme erstellt.",
        "label_no_plan": "Für diesen Bericht wurde kein 4-Wochen-Plan erstellt.",
        "label_no_general_products": "Keine allgemeinen Produktempfehlungen verfügbar.",
        "label_no_system_cards": "Für diesen Bericht wurden keine Systemkarten erstellt.",
    },

    # ---------------- RU ----------------
    "ru": {
        "report_title": "Личный отчёт о благополучии",
        "complaint_report_title": "Отчёт о благополучии на основе жалоб",
        "section_personal_data": "Личные данные",
        "section_body_form": "Оценка состава тела",
        "section_general_products": "Общие рекомендации по продуктам",
        "section_46_cards": "46 системных карточек",
        "section_medical_disclaimer": "Медицинский отказ от ответственности",

        "label_sections": "Разделы",
        "label_detail_level": "Уровень детализации",

        "label_name": "Имя и фамилия",
        "label_age": "Возраст",
        "label_gender": "Пол",
        "label_height": "Рост",
        "label_weight": "Вес",
        "label_test_date": "Дата теста",

        "label_general_tag": "Общий ярлык",
        "label_form_score": "Оценка формы",
        "label_ratio": "Соотношение",
        "label_bmr": "Базовый обмен (BMR)",

        "label_status": "Состояние",
        "label_symptoms": "Симптомы",
        "label_risks": "Риски",
        "label_lifestyle": "Рекомендации по образу жизни",
        "label_product_recommendations": "Рекомендации по продуктам",
        "label_product_usage": "Использование продукта",
        "label_detailed_explanation": "Подробное объяснение",

        "badge_46_cards_hint": "Каждая из 46 карточек соответствует отдельной системе организма.",

        "label_no_risk_list": "Для этого отчёта не был сформирован список систем высокого риска.",
        "label_no_plan": "Для этого отчёта не был сформирован 4-недельный план.",
        "label_no_general_products": "Общие рекомендации по продуктам отсутствуют.",
        "label_no_system_cards": "Для этого отчёта не были сформированы системные карточки.",
    },

    # ---------------- AR ----------------
    "ar": {
        "report_title": "تقرير العافية الشخصي",
        "complaint_report_title": "تقرير العافية القائم على الشكاوى",
        "section_personal_data": "البيانات الشخصية",
        "section_body_form": "تقييم تركيب الجسم",
        "section_general_products": "التوصيات العامة بالمنتجات",
        "section_46_cards": "٤٦ بطاقة نظام",
        "section_medical_disclaimer": "بيان المسؤولية الطبية",

        "label_sections": "الأقسام",
        "label_detail_level": "مستوى التفصيل",

        "label_name": "الاسم واللقب",
        "label_age": "العمر",
        "label_gender": "الجنس",
        "label_height": "الطول",
        "label_weight": "الوزن",
        "label_test_date": "تاريخ الفحص",

        "label_general_tag": "التوصيف العام",
        "label_form_score": "درجة اللياقة",
        "label_ratio": "النسبة",
        "label_bmr": "معدل الأيض الأساسي (BMR)",

        "label_status": "الحالة",
        "label_symptoms": "الأعراض",
        "label_risks": "المخاطر",
        "label_lifestyle": "توصيات نمط الحياة",
        "label_product_recommendations": "توصيات المنتجات",
        "label_product_usage": "طريقة استخدام المنتجات",
        "label_detailed_explanation": "شرح تفصيلي",

        "badge_46_cards_hint": "كل واحدة من البطاقات الـ٤٦ تمثل نظامًا مختلفًا في الجسم.",

        "label_no_risk_list": "لم يتم إنشاء قائمة بالأنظمة عالية الخطورة لهذا التقرير.",
        "label_no_plan": "لم يتم إنشاء خطة لمدة ٤ أسابيع لهذا التقرير.",
        "label_no_general_products": "لا توجد توصيات عامة محددة بالمنتجات.",
        "label_no_system_cards": "لم يتم إنشاء بطاقات الأنظمة لهذا التقرير.",
    },

    # ---------------- FA (Farsça) ----------------
    "fa": {
        "report_title": "گزارش تندرستی شخصی",
        "complaint_report_title": "گزارش تندرستی بر اساس شکایات",
        "section_personal_data": "اطلاعات شخصی",
        "section_body_form": "ارزیابی ترکیب بدن",
        "section_general_products": "توصیه‌های عمومی محصولات",
        "section_46_cards": "۴۶ کارت سیستم",
        "section_medical_disclaimer": "بیانیه مسئولیت پزشکی",

        "label_sections": "بخش‌ها",
        "label_detail_level": "سطح جزئیات",

        "label_name": "نام و نام خانوادگی",
        "label_age": "سن",
        "label_gender": "جنسیت",
        "label_height": "قد",
        "label_weight": "وزن",
        "label_test_date": "تاریخ تست",

        "label_general_tag": "برچسب کلی",
        "label_form_score": "امتیاز فرم",
        "label_ratio": "نسبت",
        "label_bmr": "نرخ متابولیسم پایه (BMR)",

        "label_status": "وضعیت",
        "label_symptoms": "علائم",
        "label_risks": "ریسک‌ها",
        "label_lifestyle": "توصیه‌های سبک زندگی",
        "label_product_recommendations": "توصیه‌های محصول",
        "label_product_usage": "نحوه مصرف محصولات",
        "label_detailed_explanation": "توضیح کامل",

        "badge_46_cards_hint": "هر یک از ۴۶ کارت مربوط به یک سیستم متفاوت در بدن است.",

        "label_no_risk_list": "برای این گزارش فهرست سیستم‌های پرخطر تولید نشد.",
        "label_no_plan": "برای این گزارش برنامه ۴ هفته‌ای تولید نشد.",
        "label_no_general_products": "توصیه‌ی عمومی مشخصی برای محصولات وجود ندارد.",
        "label_no_system_cards": "برای این گزارش کارت‌های سیستم تولید نشد.",
    },

    # ---------------- ID (Endonezce) ----------------
    "id": {
        "report_title": "Laporan kebugaran pribadi",
        "complaint_report_title": "Laporan kebugaran berbasis keluhan",
        "section_personal_data": "Data pribadi",
        "section_body_form": "Evaluasi komposisi tubuh",
        "section_general_products": "Rekomendasi produk umum",
        "section_46_cards": "46 kartu sistem",
        "section_medical_disclaimer": "Pernyataan tanggung jawab medis",

        "label_sections": "Bagian",
        "label_detail_level": "Tingkat detail",

        "label_name": "Nama lengkap",
        "label_age": "Usia",
        "label_gender": "Jenis kelamin",
        "label_height": "Tinggi badan",
        "label_weight": "Berat badan",
        "label_test_date": "Tanggal tes",

        "label_general_tag": "Label umum",
        "label_form_score": "Skor bentuk tubuh",
        "label_ratio": "Rasio",
        "label_bmr": "Laju metabolisme basal (BMR)",

        "label_status": "Kondisi",
        "label_symptoms": "Gejala",
        "label_risks": "Risiko",
        "label_lifestyle": "Rekomendasi gaya hidup",
        "label_product_recommendations": "Rekomendasi produk",
        "label_product_usage": "Penggunaan produk",
        "label_detailed_explanation": "Penjelasan rinci",

        "badge_46_cards_hint": "Setiap dari 46 kartu mewakili satu sistem tubuh yang berbeda.",

        "label_no_risk_list": "Tidak ada daftar sistem berisiko tinggi yang dihasilkan untuk laporan ini.",
        "label_no_plan": "Tidak ada rencana 4 minggu yang dihasilkan untuk laporan ini.",
        "label_no_general_products": "Tidak ada rekomendasi umum produk yang tersedia.",
        "label_no_system_cards": "Tidak ada kartu sistem yang dihasilkan untuk laporan ini.",
    },

    # ---------------- MN (Moğolca) ----------------
    "mn": {
        "report_title": "Хувийн WELLNESS тайлан",
        "complaint_report_title": "Гомдолд суурилсан WELLNESS тайлан",
        "section_personal_data": "Хувийн мэдээлэл",
        "section_body_form": "Биеийн бүтцийн үнэлгээ",
        "section_general_products": "Ерөнхий бүтээгдэхүүний зөвлөмж",
        "section_46_cards": "46 системийн карт",
        "section_medical_disclaimer": "Эрүүл мэндийн хариуцлагын мэдэгдэл",

        "label_sections": "Хэсгүүд",
        "label_detail_level": "Дэлгэрэнгүй түвшин",

        "label_name": "Нэр",
        "label_age": "Нас",
        "label_gender": "Хүйс",
        "label_height": "Өндөр",
        "label_weight": "Жин",
        "label_test_date": "Шинжилгээний огноо",

        "label_general_tag": "Ерөнхий тэмдэглэгээ",
        "label_form_score": "Формын оноо",
        "label_ratio": "Харьцаа",
        "label_bmr": "Суурь бодисын солилцоо (BMR)",

        "label_status": "Төлөв",
        "label_symptoms": "Шинж тэмдгүүд",
        "label_risks": "Эрсдэлүүд",
        "label_lifestyle": "Амьдралын хэв маягийн зөвлөмж",
        "label_product_recommendations": "Бүтээгдэхүүний зөвлөмж",
        "label_product_usage": "Бүтээгдэхүүн хэрэглэх заавар",
        "label_detailed_explanation": "Дэлгэрэнгүй тайлбар",

        "badge_46_cards_hint": "46 карт тус бүр нь биеийн өөр системийг илэрхийлнэ.",

        "label_no_risk_list": "Энэ тайланд өндөр эрсдэлтэй системийн жагсаалт үүсгээгүй байна.",
        "label_no_plan": "Энэ тайланд 4 долоо хоногийн төлөвлөгөө үүсгээгүй байна.",
        "label_no_general_products": "Ерөнхий түвшний бүтээгдэхүүний тодорхой зөвлөмж байхгүй.",
        "label_no_system_cards": "Энэ тайланд системийн карт үүсгээгүй байна.",
    },
}


def build_labels(target_lang: str, second_lang: str) -> Dict[str, str]:
    """
    Rapor içi başlık ve etiket metinlerini hazırlar.
    - 1. dil (target_lang) esastır.
    - 2. dil varsa, "TR / EN" mantığı ile tek satırda birleştirilir.
    """
    primary = (target_lang or "tr").lower()
    if primary not in LABEL_TEXTS:
        primary = "tr"

    second = (second_lang or "").lower()
    if second not in LABEL_TEXTS:
        second = ""

    base_primary = LABEL_TEXTS[primary]

    # Tek dilli
    if not second:
        return base_primary

    base_second = LABEL_TEXTS[second]

    combined: Dict[str, str] = {}
    all_keys = set(base_primary.keys()) | set(base_second.keys())
    for key in all_keys:
        p = base_primary.get(key, "")
        s = base_second.get(key, "")
        if p and s:
            combined[key] = f"{p} / {s}"
        else:
            combined[key] = p or s
    return combined

# ====================================================
#  GEÇMİŞ RAPOR KAYDI
# ====================================================

HISTORY_FILE = os.path.join("tmp", "history.json")
HISTORY_MAX_DAYS = 90  # son 90 günü sakla


def load_history() -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_history_entry(entry: Dict[str, Any]) -> None:
    os.makedirs("tmp", exist_ok=True)
    history = load_history()

    # created_at yoksa ekle
    now = datetime.utcnow()
    if "created_at" not in entry or not entry.get("created_at"):
        entry["created_at"] = now.isoformat()

    history.append(entry)

    # Son 90 günden eski kayıtları süz
    cutoff = now - timedelta(days=HISTORY_MAX_DAYS)
    filtered: List[Dict[str, Any]] = []
    for item in history:
        ts = item.get("created_at")
        dt = None
        if isinstance(ts, str):
            # "2025-12-10T10:20:30" / "2025-12-10T10:20:30Z" ikisini de tolere et
            try:
                dt = datetime.fromisoformat(ts.replace("Z", ""))
            except Exception:
                dt = None
        # Tarihi hiç parse edemezsek yine de sileyim demeyelim, dursun
        if (dt is None) or (dt >= cutoff):
            filtered.append(item)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)


def get_history_entry(report_id: str) -> Dict[str, Any] | None:
    history = load_history()
    for item in history:
        if item.get("id") == report_id:
            return item
    return None

import uuid  # en üst import'lara ekleyebilirsin

def create_report_id(prefix: str = "rpt") -> str:
    """
    Hem PDF, hem şikayet, hem karşılaştırma raporları için
    benzersiz rapor ID'si üretir.
    Örnek: cmp_20251207_ABC123
    """
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    short = uuid.uuid4().hex[:6].upper()
    return f"{prefix}_{ts}_{short}"


# ====================================================
#  YARDIMCI FONKSİYONLAR
# ====================================================

def strip_device_explanations(raw_text: str) -> str:
    """
    Cihaz PDF'lerinden gelen metinde sadece işimize yarayan kısımları bırakır:
    - Kart başlıkları
    - Kişi bilgileri
    - 'Gerçek Test Sonuçları' tablosu

    'Parametre Açıklaması' (ve diğer dillerdeki karşılıkları) ile başlayan
    uzun açıklama bloklarını ve
    'Test sonuçları yalnızca referans amaçlıdır...' (ve diğer diller) satırına
    kadar olan kısmı metinden çıkarır.

    Desteklenen diller:
    - Türkçe
    - İngilizce
    - Almanca
    - İspanyolca
    - Rusça
    """

    if not raw_text:
        return ""

    lines = raw_text.splitlines()
    keep_lines = []
    skip = False

    # Açıklama bloğunun BAŞLANGIÇ anahtar kelimeleri
    start_markers = [
        # Türkçe
        "Parametre Açıklaması",
        "Parametre açıklaması",

        # İngilizce
        "Parameter Explanation",
        "Explanation of Parameters",
        "Parameter description",

        # Almanca
        "Parametererklärung",
        "Erläuterung der Parameter",

        # İspanyolca
        "Explicación de los parámetros",
        "Explicacion de los parametros",

        # Rusça
        "Объяснение параметров",
        "Пояснение параметров",
    ]

    # Açıklama bloğunun BİTİŞ / UYARI anahtar kelimeleri
    end_markers = [
        # Türkçe
        "Test sonuçları yalnızca referans amaçlıdır",
        "Test sonuçları sadece referans amaçlıdır",

        # İngilizce
        "Test results are for reference only",
        "The test results are for reference only",

        # Almanca
        "Testergebnisse dienen nur als Referenz",
        "Die Testergebnisse dienen nur als Referenz",

        # İspanyolca
        "Los resultados de la prueba son solo de referencia",
        "Los resultados del test son solo de referencia",

        # Rusça
        "Результаты теста предназначены только для справки",
        "Результаты исследования предназначены только для справки",
    ]

    def has_marker(s: str, markers: list[str]) -> bool:
        s_low = s.lower()
        for m in markers:
            if m.lower() in s_low:
                return True
        return False

    for line in lines:
        stripped = line.strip()

        # Açıklama bloğu başlıyor → bundan sonrasını (şimdilik) alma
        if has_marker(stripped, start_markers):
            skip = True
            continue

        # Uyarı / referans cümlesi → buradan sonra tekrar almaya başla
        if has_marker(stripped, end_markers):
            skip = False
            continue

        if not skip:
            keep_lines.append(line)

    return "\n".join(keep_lines)


def read_pdf_text(file_path: str) -> str:
    """
    PDF'teki tüm sayfaları okur, metni birleştirir ve
    cihazın uzun açıklama / uyarı bloklarını temizler.
    (Tablolar + kişi bilgileri + kart başlıkları korunur.)
    """
    reader = PdfReader(file_path)
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue

    raw = "\n".join(parts)
    cleaned = strip_device_explanations(raw)
    return cleaned


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_vucut_formu_from_device_pdf(pdf_text: str) -> Dict[str, Any]:
    text = pdf_text
    result: Dict[str, Any] = {
        "etiket": "",
        "oran": "",
        "aciklama": "",
        "bmr_kcal": "",
        "form_puani": "",
    }

    bmr_pattern = re.compile(
        r"Bazal\s+metabolizma\s+h[ıi]z[ıi]\s*\(BMR\)\s*([0-9]+)",
        re.IGNORECASE,
    )
    m_bmr = bmr_pattern.search(text)
    if m_bmr:
        result["bmr_kcal"] = m_bmr.group(1)

    form_pattern = re.compile(
        r"V[üu]cut\s*formu\s*De[ğg]erlendirmesi\s*[:=]\s*([0-9]+[.,]?[0-9]*)",
        re.IGNORECASE,
    )
    m_form = form_pattern.search(text)
    if m_form:
        raw = m_form.group(1).replace(",", ".")
        try:
            score = float(raw)
            result["form_puani"] = f"{score:.1f}".replace(".", ",")
        except ValueError:
            pass

    explanation = ""
    if result["form_puani"]:

        explanation = (
            "Vücut formu değerlendirme puanınız "
            f"{result['form_puani']}/100 civarındadır. "
            "Genel olarak kas-kemik yapısı, yağ oranı ve vücut kompozisyonunuz hakkında bilgi verir. "
            "Skoru iyileştirmek için dengeli beslenme, düzenli hareket ve kaliteli uyku önemlidir."
        )

    if result["bmr_kcal"]:
        explanation += (
            "\n\nBazal metabolizma hızınız (BMR) yaklaşık "
            f"{result['bmr_kcal']} kcal/gün’dür. "
            "Bu değer; hiçbir aktivite yapmasanız bile vücudunuzun temel fonksiyonları "
            "için ihtiyaç duyduğu enerji miktarını ifade eder."
        )

    if explanation:
        result["aciklama"] = explanation

    if result["form_puani"]:
        try:
            score = float(result["form_puani"].replace(",", "."))
            if score >= 70:
                result["etiket"] = "normal"
            elif 60 <= score < 70:
                result["etiket"] = "geliştirilmeli"
            else:
                result["etiket"] = "riskli"
        except Exception:
            pass

    if result["form_puani"] and not result["oran"]:
        result["oran"] = result["form_puani"] + "%"

    if not any(v for v in result.values()):
        return {}
    return result


def build_language_instruction(target_lang: str, second_lang: str, source: str) -> str:
    if target_lang == "auto":
        if source == "pdf":
            base = (
                "Raporun birincil dilini PDF metninin diline göre belirle. "
                "PDF Türkçe ise çıktıyı Türkçe yaz; değilse PDF'den tespit ettiğin dilde yaz."
            )
        else:
            base = (
                "Raporun birincil dilini verilen metnin diline göre belirle. "
                "Metin Türkçe ise çıktıyı Türkçe yaz; değilse metinden tespit ettiğin dilde yaz."
            )
        primary_label = "metnin algılanan dili"
    else:
        primary_label = LANGUAGE_LABELS.get(target_lang, target_lang)
        base = (
            f"Tüm rapor metinlerinin BİRİNCİL dilini sadece {primary_label} olarak kullan. "
            "Rapor metinlerinde birincil dil dışında farklı bir dil kullanma."
        )

    if second_lang:
        second_label = LANGUAGE_LABELS.get(second_lang, second_lang)
        base += (
            f"\n\nİKİ DİLLİ FORMAT ZORUNLUDUR: İkinci dil seçili ({second_label}). "
            "Bu durumda TÜM METİN ALANLARINI iki satırlı yaz:\n"
            "- 1. satır: SADECE birincil dil metni.\n"
            "- 2. satır: SADECE ikinci dil çevirisi.\n\n"
            "Bu kural ŞU TÜM STRING ALANLAR İÇİN GEÇERLİDİR:\n"
            "- vucut_formu.aciklama\n"
            "- genel_bulgu.ozet\n"
            "- genel_bulgu.en_riskli_10_sistem[*].sorun_ozeti\n"
            "- genel_bulgu.onemore_urun_onerileri[*].neden\n"
            "- dort_haftalik_plan[*].detay\n"
            "- dort_haftalik_plan[*].urun_kullanimi\n"
            "- sistem_kartlari[*].sistem_adi, durum, belirtiler, riskler, yasam_tavsiyesi\n"
            "- sistem_kartlari[*].urun_onerileri[*].neden\n"
            "- kanallar_ve_kollateraller_detay içindeki tüm açıklama alanları\n"
            "- insan_bilinc_duzeyi_detay içindeki tüm açıklama alanları\n"
            "- şikayet bazlı rapordaki tüm açıklama alanları\n"
            "- karşılaştırma raporundaki tüm açıklama alanları\n"
            "- tüm tibbi_sorumluluk alanları\n\n"
            "ÖRNEK STRING DEĞERİ:\n"
            "\"ozet\": \"Birincil dil cümlesi.\\nSecond language sentence.\"\n"
        )
    else:
        base += (
            "\nİkinci dil seçili değil, sadece tek dilli metin üret."
        )

    return base


def build_analysis_prompt(
    pdf_text: str,
    detail_level: DetailLevel,
    target_lang: str,
    second_lang: str,
    brand: str,
) -> str:
    # ÇIKTIYI KISA TUTMA TALİMATI
    level_desc = (
        "ULTRA DETAY: Öğretici ve anlaşılır ol, tıbbi metin yazarsan yanına parantez içinde açıklamasını yaz, ancak METNİ KISA TUT.\n"
        "- Her alanda mümkün olduğunca 2-3 cümleyi geçme.\n"
        "- Genel bulgu bölümünde toplam 25-40 cümleyi geçme.\n"
        "- 4 haftalık planda her hafta için 2-3 cümle yaşam tarzı, 2-3 cümle ürün kullanımı yaz.\n"
        "- Kanallar & kollateraller ve İnsan Bilinç Düzeyi açıklamalarını 3-4 cümleyle sınırla.\n"
        "- Her sistem kartında (durum, belirtiler, riskler, yaşam tavsiyesi) alan başına en fazla 2 cümle yaz.\n"
        "- Toplam çıktı çok uzun olmamalı; gerekirse bazı açıklamaları TEK CÜMLEYE indir."
    )

    system_block = "\n".join(f"- Kart {i+1}: {name}" for i, name in enumerate(SYSTEM_NAMES))

    brand_label = BRAND_LABELS.get(brand, "OneMore International")
    products = get_brand_products(brand)
    products_block = "\n".join(f"- {p}" for p in products)
    lang_instruction = build_language_instruction(target_lang, second_lang, source="pdf")

    brand_rules = (
        "KESİN ÜRÜN & MARKA KURALLARI:\n"
        "1) ÜRÜN ÖNERİLERİ sadece ve sadece ÇALIŞILAN MARKA listesinden seçilecek. "
        "   Hiçbir şekilde başka marka ismi yazma.\n"
        "2) Marka 'Genel wellness analizi (markasız)' ise marka ismi hiç kullanma, sadece ürün tipi yaz.\n"
        "3) Aynı ürünü 46 sistem kartının tamamında tekrar etme; bir ürünü en fazla 6-8 kartta kullan.\n"
        "4) Cinsiyet bilgisi uygunsuzsa (erkek için kadın ürünü veya tersi) o ürünü yazma.\n"
        "5) Her ürün için 'neden' alanında o sistemi hedefleyen net ve spesifik iki cümle yaz.\n"
    )

    prompt = f"""
❗ JSON çıktısı ZORUNLUDUR.
❗ Üretilen çıktı TAMAMEN VE SADECE JSON olacaktır.
❗ JSON içindeki HER string alanı (tırnaklar dahil) eksiksiz kapatılacaktır.
❗ YANIT YARIDA KESİLMEYECEKTİR.
❗ Eğer içerik çok uzunsa, METNİ KISALT ama JSON YAPISINI ASLA BOZMA.
❗ JSON DIŞINDA tek bir kelime bile yazma (Markdown, açıklama, serbest metin YASAK).

Aşağıda bir Wellness Analyzer cihazından alınmış test PDF metni var.
Bu metni kullanarak kişiye özel, anlaşılır ve tıbbi teşhis içermeyen
BİR WELLNESS RAPORU hazırla.

ÇALIŞILAN MARKA:
- Marka adı: {brand_label}
- Ürün önerilerini SADECE bu markanın ürünlerinden seç.
- FARKLI MARKALARA AİT ÜRÜN İSİMLERİNİ ASLA KULLANMA.
- Aşağıdaki ürün listesi dışına ÇIKMA, yeni ürün uydurma.

DİL TALİMATI:
{lang_instruction}

{brand_rules}

VÜCUT FORMU:
- 'vucut_formu.etiket': kısa özet (zayıf/normal/fazla kilolu/obez vb.).
- 'vucut_formu.oran': form skoru veya aralık.
- 'vucut_formu.aciklama': 1-2 cümlelik özet yorum.
- 'vucut_formu.bmr_kcal': BMR.
- 'vucut_formu.form_puani': 0-100 arası puan.

GENEL BULGU:
- 'genel_bulgu.ozet' DOLU olsun (yaklaşık 25-40 cümle).
- 'genel_bulgu.en_riskli_10_sistem' TAM 10 eleman içersin (her biri 3 kısa cümle).
- 'genel_bulgu.onemore_urun_onerileri' 8-12 ürün içersin (her biri kısa neden ve süre).

4 HAFTALIK YAŞAM PLANI:
- 'dort_haftalik_plan' mutlaka 4 eleman (hafta 1-4) içersin.
- Her elemanda 'hafta', 'odak', 'detay', 'urun_kullanimi' DOLU olsun.
- 'detay' alanı en fazla 2-3 cümle, 'urun_kullanimi' en fazla 2-3 cümle olsun.

KANALLAR & KOLLATERALLER DETAY:
- Tüm metin alanlarını DOLDUR (boş bırakma).
- Her metin alanında 3-4 cümleyi geçme.

İNSAN BİLİNÇ DÜZEYİ DETAY:
- Tüm metin alanlarını DOLDUR (boş bırakma).
- Her metin alanında 3-4 cümleyi geçme.

SİSTEM KARTLARI (EN KRİTİK KISIM):
- Aşağıdaki 46 sistemin HER BİRİ için mutlaka bir kart üret:
{system_block}

- TAM 46 sistem kartı üret; hiçbir sistemi atlama.
- Her kartta ŞU ALANLAR ZORUNLUDUR:
  'sistem_adi', 'durum', 'belirtiler', 'riskler', 'yasam_tavsiyesi', 'urun_onerileri'.
- HİÇBİR kartı tamamen boş bırakma; her kartta
  'durum', 'belirtiler', 'riskler', 'yasam_tavsiyesi' alanları DOLU olsun.
- Her bir metin alanında en fazla 2 cümle yaz (kısa ama net).
- 'urun_onerileri' listesinde her sistem için 2-3 ürün yazmaya çalış.
- Her karttaki açıklamalar KİŞİNİN TEST SONUÇLARINA UYGUN olsun;
  sadece genel, kopyala-yapıştır metinler yazma.
  SİSTEM KARTLARI (EN KRİTİK KISIM):
- Aşağıdaki 46 sistemin HER BİRİ için mutlaka bir kart üret:
{system_block}

- TAM 46 sistem kartı üret; hiçbir sistemi atlama.
- Her kartta ŞU ALANLAR ZORUNLUDUR:
  'sistem_adi', 'durum', 'belirtiler', 'riskler', 'yasam_tavsiyesi', 'urun_onerileri'.
- 'urun_onerileri' ALANI MUTLAKA BİR LİSTE OLACAKTIR ve şu yapıda doldurulacaktır:
  "urun_onerileri": [
    {
      "urun": "omega-3 takviyesi",
      "neden": "Bu sistemdeki bulgulara yönelik, neden bu tip desteğin uygun olduğunu açıklayan 1-2 cümle.",
      "sure": "4-8 hafta aralığında net süre (örneğin '8 hafta')."
    },
    {
      "urun": "C vitamini takviyesi",
      "neden": "Bağışıklık, oksidatif stres veya ilgili bulgulara uygun neden açıklaması.",
      "sure": "4 hafta"
    }
  ]
- Marka 'Genel wellness analizi (markasız)' ise:
  - 'urun' alanlarında MARKA ADI KULLANMA; sadece ürün TİPİ veya KATEGORİ yaz 
    (örnek: "probiyotik takviyesi", "bitkisel detoks çayı", "multivitamin + mineral kompleksi").
- Hiçbir kartı tamamen boş bırakma; her kartta
  'durum', 'belirtiler', 'riskler', 'yasam_tavsiyesi' alanları DOLU olsun.
- Her bir metin alanında en fazla 2 cümle yaz (kısa ama net).
- 'urun_onerileri' listesinde her sistem için mümkünse 2-3 ürün yazmaya çalış.
- Her karttaki açıklamalar KİŞİNİN TEST SONUÇLARINA UYGUN olsun;
  sadece genel, kopyala-yapıştır metinler yazma.


PDF METNİ:
------------------
{pdf_text}
------------------

{level_desc}

ÇIKTI FORMATIN (SADECE JSON):

{{
  "kisi_bilgileri": {{
    "ad_soyad": "",
    "yas": "",
    "cinsiyet": "",
    "boy_cm": "",
    "kilo_kg": "",
    "test_tarihi": ""
  }},
  "vucut_formu": {{
    "etiket": "",
    "oran": "",
    "aciklama": "",
    "bmr_kcal": "",
    "form_puani": ""
  }},
  "genel_bulgu": {{
    "ozet": "",
    "en_riskli_10_sistem": [
      {{
        "sistem_adi": "",
        "sorun_ozeti": ""
      }}
    ],
    "onemore_urun_onerileri": [
      {{
        "urun": "",
        "neden": "",
        "sure": ""
      }}
    ]
  }},
  "dort_haftalik_plan": [
    {{
      "hafta": 1,
      "odak": "",
      "detay": "",
      "urun_kullanimi": ""
    }}
  ],
  "kanallar_ve_kollateraller_detay": {{
    "durum": "",
    "belirtiler": "",
    "riskler": "",
    "yasam_onerileri": "",
    "detayli_aciklama": "",
    "onemore_urun_onerileri": [
      {{
        "urun": "",
        "neden": "",
        "sure": ""
      }}
    ]
  }},
  "insan_bilinc_duzeyi_detay": {{
    "durum": "",
    "belirtiler": "",
    "riskler": "",
    "yasam_onerileri": "",
    "detayli_aciklama": "",
    "onemore_urun_onerileri": [
      {{
        "urun": "",
        "neden": "",
        "sure": ""
      }}
    ]
  }},
  "sistem_kartlari": [
    {{
      "sistem_adi": "",
      "durum": "",
      "belirtiler": "",
      "riskler": "",
      "yasam_tavsiyesi": "",
      "urun_onerileri": [
        {{
          "urun": "",
          "neden": "",
          "sure": ""
        }}
      ]
    }}
  ],
  "tibbi_sorumluluk": ""
}}

{brand_label} ürünlerini yalnızca şu listeden seç (isimleri birebir kullan):

{products_block}

SADECE GEÇERLİ JSON DÖN; açıklama yazma, Markdown veya ``` kullanma.
"""
    return prompt


def build_analysis_prompt_part1(
    pdf_text: str,
    detail_level: DetailLevel,
    target_lang: str,
    second_lang: str,
    brand: str,
) -> str:
    level_desc = (
        "ULTRA DETAY: Öğretici ve anlaşılır ol.\n"
        "- GENEL BULGU bölümünde, 46 sistemin genel bir özetini de kapsayacak şekilde ZENGİN içerik üret.\n"
        "- Genel bulgu özetinde toplamda yaklaşık 30–60 cümle aralığında kal; çok kısa yazma.\n"
        "- 4 haftalık planda her hafta için 2–4 cümle yaşam tarzı, 2–3 cümle ürün kullanımı yaz.\n"
        "- Kanallar & kollateraller ve İnsan Bilinç Düzeyi açıklamalarını 4–6 cümleyle sınırla.\n"
        "- Sistem kartları bu çağrıda üretilmeyecek, sadece genel resim ver."
    )

    brand_label = BRAND_LABELS.get(brand, "OneMore International")
    products = get_brand_products(brand)
    products_block = "\n".join(f"- {p}" for p in products)
    lang_instruction = build_language_instruction(target_lang, second_lang, source="pdf")

    brand_rules = (
        "KESİN ÜRÜN & MARKA KURALLARI:\n"
        "1) ÜRÜN ÖNERİLERİ sadece ve sadece ÇALIŞILAN MARKA listesinden seçilecek. "
        "   Hiçbir şekilde başka marka ismi yazma.\n"
        "2) Marka 'Genel wellness analizi (markasız)' ise marka ismi hiç kullanma, sadece ürün tipi yaz.\n"
        "3) Aynı ürünü tüm raporda aşırı tekrar etme; mantıklı dağıt ama yine de listeyi doldur.\n"
        "4) Cinsiyet bilgisi uygunsuzsa (erkek için kadın ürünü veya tersi) o ürünü yazma.\n"
        "5) Her ürün için 'neden' alanında o alanı hedefleyen net ve spesifik bir cümle yaz.\n"
        "6) 'urun' alanı BOŞ KALAMAZ; mutlaka aşağıdaki listeden BİREBİR isim yaz.\n"
    )

    # TIBBİ SORUMLULUK BEYANI İÇİN KURAL
    # - Tıbbi sorumluluk beyanında ASLA marka adı, cihaz adı kullanma.
    # - "Bu rapor" veya "bu değerlendirme" gibi nötr ifadeler kullan.
    # - OneMore, Xaura, WellnessAnalyzer, cihaz adı vb. MARKA VE CİHAZ isimleri YAZMA.

    prompt = f"""
❗ JSON çıktısı ZORUNLUDUR.
❗ Üretilen çıktı TAMAMEN VE SADECE JSON olacaktır.
❗ JSON içindeki HER string alanı eksiksiz ve geçerli olmalıdır.
❗ JSON DIŞINDA tek bir kelime bile yazma (Markdown, açıklama, serbest metin YASAK).

Aşağıda bir Wellness Analyzer cihazından alınmış test PDF metni var.
Bu metni kullanarak kişiye özel, anlaşılır ve tıbbi teşhis içermeyen
BİR WELLNESS RAPORUNUN SADECE 1. BÖLÜMÜNÜ hazırla.

1. BÖLÜMDE OLMASI GEREKENLER:
- kisi_bilgileri
- vucut_formu
- genel_bulgu (ozet, en_riskli_10_sistem, onemore_urun_onerileri)
- dort_haftalik_plan
- kanallar_ve_kollateraller_detay
- insan_bilinc_duzeyi_detay
- tibbi_sorumluluk

DIKKAT:
- BU BÖLÜMDE 'sistem_kartlari' ÜRETME.
- 'sistem_kartlari' alanını yazma veya BOŞ LİSTE [] bırak; detaylı 46 kart 2. bölümde gelecek.

ÇOCUK / YETİŞKİN KURALI:
- PDF metnindeki kişi bilgilerini incele ve YAŞ bilgisini tespit etmeye çalış.
- Eğer yaş 10'un ALTINDAYSA bu bir ÇOCUK RAPORUDUR.
- ÇOCUK RAPORU durumunda:
  * Genel özet, en riskli sistemler, 4 haftalık plan ve genel ürün önerilerini özellikle şu alanlara dayandır:
    - Eser / iz elementler
    - Vitaminler
    - Amino asitler
    - Koenzimler
    - Esansiyel yağ asitleri / yağ asitleri
    - ADHD / dikkat ve odaklanma
    - Ergen zekası / bilişsel gelişim
    - Ergen büyüme endeksi / büyüme potansiyeli
    - Lesitin
  * Prostat, erkek cinsel fonksiyonu, sperm ve meni, menopoz, gebelik, rahim, kadınlık hormonları,
    adet döngüsü gibi YETİŞKİN konularından bahsetme.
- Eğer yaş 10 ve ÜZERİNDEYSE, raporu YETİŞKİN mantığıyla hazırla (46 sistemi kapsayan genel bir özet yaz).

GENEL BULGU KURALLARI:
- 'genel_bulgu.ozet' mutlaka DOLU olsun ve kişinin 46 sisteminin genel durumunu anlatan zengin bir metin olsun.
- 'genel_bulgu.ozet' içinde en az 10–15 farklı sistem veya tema hakkında kısa paragraflar olsun.
- 'genel_bulgu.en_riskli_10_sistem' listesi TAM OLARAK 10 eleman içerir.
- Bu 10 elemanın HER BİRİNDE 'sistem_adi' ve 'sorun_ozeti' DOLU olacak.
- 'sistem_adi' alanında 'bulunamadı', 'yok' gibi ifadeler KULLANMA.
- Hiçbir durumda 'Bu bölümde vurgulanacak özel riskli sistem bulunamadı' gibi bir cümle yazma.
- Mutlaka 10 sistem seç ve her biri için gerçek ve anlamlı bir özet yaz.

GENEL ÜRÜN ÖNERİLERİ:
- 'genel_bulgu.onemore_urun_onerileri' listesi EN AZ 7, EN FAZLA 10 ürün içerir.
- Her üründe:
  - 'urun': Aşağıdaki ürün listesinden birebir isim.
  - 'neden': O ürünün genel bulguyu nasıl desteklediğini anlatan EN AZ 2 cümle.
  - 'sure': Örn. '4 hafta', '8 hafta' gibi bir süre bilgisi.

4 HAFTALIK YAŞAM PLANI:
- 'dort_haftalik_plan' mutlaka 4 eleman (hafta 1–4) içersin.
- Her elemanda 'hafta', 'odak', 'detay', 'urun_kullanimi' DOLU olsun.
- 'detay' alanı 3–5 cümle, 'urun_kullanimi' 2–3 cümle olsun.
- ÇOCUK ise, planı çocuğun yaşı ve gelişim dönemine uygun tut; yetişkin cinsellik, hamilelik, menopoz vb. konulara girmeden yaz.

KANALLAR & KOLLATERALLER DETAY:
- Tüm metin alanlarını DOLDUR (boş bırakma).
- Her metin alanında 3–5 cümle kullan.

İNSAN BİLİNÇ DÜZEYİ DETAY:
- Tüm metin alanlarını DOLDUR (boş bırakma).
- Her metin alanında 3–5 cümle kullan.
- ÇOCUK ise, anlatımı çocuğun gelişim düzeyine uygun, aile odaklı ve pedagojik hassasiyete sahip şekilde tut.

DİL TALİMATI:
{lang_instruction}

ÇALIŞILAN MARKA:
- Marka adı: {brand_label}
- Ürün önerilerini SADECE bu markanın ürünlerinden seç.
- Aşağıdaki ürün listesi dışına ÇIKMA, yeni ürün ismi uydurma.

PDF METNİ:
------------------
{pdf_text}
------------------

{level_desc}

ÇIKTI FORMATIN (SADECE 1. BÖLÜMÜN JSON'U):

{{
  "kisi_bilgileri": {{
    "ad_soyad": "",
    "yas": "",
    "cinsiyet": "",
    "boy_cm": "",
    "kilo_kg": "",
    "test_tarihi": ""
  }},
  "vucut_formu": {{
    "etiket": "",
    "oran": "",
    "aciklama": "",
    "bmr_kcal": "",
    "form_puani": ""
  }},
  "genel_bulgu": {{
    "ozet": "",
    "en_riskli_10_sistem": [
      {{
        "sistem_adi": "",
        "sorun_ozeti": ""
      }}
    ],
    "onemore_urun_onerileri": [
      {{
        "urun": "",
        "neden": "",
        "sure": ""
      }}
    ]
  }},
  "dort_haftalik_plan": [
    {{
      "hafta": 1,
      "odak": "",
      "detay": "",
      "urun_kullanimi": ""
    }}
  ],
  "kanallar_ve_kollateraller_detay": {{
    "durum": "",
    "belirtiler": "",
    "riskler": "",
    "yasam_onerileri": "",
    "detayli_aciklama": "",
    "onemore_urun_onerileri": [
      {{
        "urun": "",
        "neden": "",
        "sure": ""
      }}
    ]
  }},
  "insan_bilinc_duzeyi_detay": {{
    "durum": "",
    "belirtiler": "",
    "riskler": "",
    "yasam_onerileri": "",
    "detayli_aciklama": "",
    "onemore_urun_onerileri": [
      {{
        "urun": "",
        "neden": "",
        "sure": ""
      }}
    ]
  }},
  "tibbi_sorumluluk": ""
}}

{brand_rules}

{brand_label} ürünlerini yalnızca şu listeden seç (isimleri birebir kullan):

{products_block}

SADECE GEÇERLİ JSON DÖN; açıklama yazma, Markdown veya ``` kullanma.
"""
    return prompt


def build_system_cards_prompt(
    pdf_text: str,
    target_lang: str,
    second_lang: str,
    brand: str,
) -> str:
    brand_label = BRAND_LABELS.get(brand, "OneMore International")
    products = get_brand_products(brand)
    products_block = "\n".join(f"- {p}" for p in products)

    # Mevcut dil talimatını yine kullanalım
    lang_instruction = build_language_instruction(target_lang, second_lang, source="pdf")

    # Cihazın orijinal sistem isimleri (Türkçe) – sadece REFERANS
    system_block = "\n".join(f"- {name}" for name in SYSTEM_NAMES)

    # 1. ve 2. dil bilgisi
    primary = (target_lang or "tr").lower().strip()
    secondary = (second_lang or "").lower().strip()
    bilingual = bool(secondary)

    if bilingual:
        extra_lang_rule = f"""
EK DİL KURALI:
- İki dilli çalışıyorsun.
- Her kartta "sistem_adi" alanını şu formatta yaz:
  "1. DİL sistem adı / 2. DİL sistem adı"
- 1. dil: {primary.upper()}
- 2. dil: {secondary.upper()}
- Aşağıdaki Türkçe sistem isimlerini ÇIKTIDA AYNI TÜRKÇE ŞEKLİNDE KULLANMA.
  Sadece anlam olarak referans al; başlığı hedef dillere çevir.
"""
    else:
        extra_lang_rule = f"""
EK DİL KURALI:
- Tek dilli çalışıyorsun.
- Her kartta "sistem_adi" alanını SADECE {primary.upper()} dilinde yaz.
- Aşağıdaki Türkçe sistem isimlerini ÇIKTIDA AYNI TÜRKÇE ŞEKLİNDE KULLANMA.
  Sadece anlam olarak referans al; başlığı hedef dile çevir.
"""

    prompt = f"""
❗ JSON çıktısı ZORUNLUDUR.
❗ Üretilen çıktı TAMAMEN VE SADECE JSON olacaktır.
❗ YANIT YARIDA KESİLMEYECEKTİR.
❗ JSON DIŞINDA tek bir kelime bile yazma.

Bu çağrıda SADECE 2. BÖLÜMÜ üret:
- 46 sistemlik detaylı sistem kartları.

DİL TALİMATI:
{lang_instruction}
{extra_lang_rule}

ÇALIŞILAN MARKA:
- Marka adı: {brand_label}
- Ürün önerilerini SADECE bu markanın ürünlerinden seç.
- Aşağıdaki ürün listesi dışına çıkma, yeni ürün ismi uydurma.

PDF METNİ:
------------------
{pdf_text}
------------------

ORİJİNAL CİHAZ SİSTEM LİSTESİ (TÜRKÇE REFERANS, ÇIKTIDA AYNEN KULLANMA):
{system_block}

46 SİSTEM İÇİN KURALLAR:
- TAM OLARAK 46 sistem kartı üret; hiçbir sistemi atlama.
- Kart sırası, yukarıdaki sistem isimlerinin sırasına uygun olsun (ama başlıkları hedef dil/dillerde yaz).
- Her kartta ZORUNLU alanlar:
  "sistem_adi", "durum", "belirtiler", "riskler", "yasam_tavsiyesi", "urun_onerileri".
- Hiçbir kart için "boş", "bulunamadı" vb. ifadeler kullanma; mutlaka anlamlı metin üret.
- Her alanda KISA ama NET 1–2 cümle kullan.
- 'sistem_adi' alanına:
  * İki dilli ise: "1. dil sistem adı / 2. dil sistem adı" formatında yaz.
  * Tek dilli ise: yalnızca hedef dilde sistem başlığı yaz.
  * Numara veya 'Kart 19' vb. ekleme.
- 'urun_onerileri' her kartta mümkünse 2–3 ürün içersin.
- Her üründe:
  - 'urun': Aşağıdaki ürün listesinden birebir isim.
  - 'neden': O sistem için en az 1–2 cümlelik, spesifik ve tekrarsız açıklama.
  - 'sure': Örn. '4 hafta'.

ÇOK ÖNEMLİ:
- Aynı cümleyi 46 kartta kopyala-yapıştır yapma.
- Metinler benzer tema içerebilir ama her kartta en azından 1–2 ifade o sisteme özgü olsun.
- 'Bu sistem genel olarak...' gibi tamamen generic kalıplarla yetinme; sistem adını ve işlevini referans al.

ÇIKTI ŞEMASI:

{{
  "sistem_kartlari": [
    {{
      "sistem_adi": "",
      "durum": "",
      "belirtiler": "",
      "riskler": "",
      "yasam_tavsiyesi": "",
      "urun_onerileri": [
        {{
          "urun": "",
          "neden": "",
          "sure": ""
        }}
      ]
    }}
  ]
}}

{brand_label} ürünlerini yalnızca şu listeden seç (isimleri birebir kullan):

{products_block}

SADECE GEÇERLİ JSON DÖN; açıklama yazma, Markdown veya ``` kullanma.
"""
    return prompt

# ============================================================
#  ŞİKAYET ANALİZİ İÇİN MARKA BAZLI ÜRÜN LİSTESİ + HELPER'LAR
#  (BU BLOĞU ESKİ COMPLAINT_BRAND_PRODUCTS / apply_* YERİNE KOY)
# ============================================================

COMPLAINT_BRAND_PRODUCTS: dict[str, list[str]] = {
   
    "onemore": [
        "OneMore Painless Night Glu plus+",
        "OneMore Slim Style",
        "OneMore Painless Night Glu",
        "OneMore Fitmore Shake",
        "OneMore B12 Plus",
        "OneMore Glutamore",
        "OneMore Omevia",
        "OneMore Dekamin",
        "OneMore Melatoninplus",
        "OneMore Lady (Kadın)",
        "OneMore Gentleman (Erkek)",
        "OneMore Omicoff Coffee",
    ],

    # =========================
    # Xaura Global
    # =========================
    "xaura": [
        "XAura Pain",
        "XAura X-12",
        "xaura X-DK",
        "xaura X-LIM",
        "xaura X-RECOVERY",
        "xaura X-HE (ERKEK)",
        "xaura X-SHE (KADIN)",
        "xaura X-OMEGA",
        "xaura X-NIGHT",
        "xaura X-COL",
        "xaura Coffee Reishi",
    ],

    # =========================
    # Atomy
    # =========================
    "atomy": [
        "Atomy Hongsamdan Ginseng",
        "Atomy Probiotics",
        "Atomy Omega 3",
        "Atomy Color Food Vitamin C",
        "Atomy Noni Pouch",
        "Atomy Spirulina",
        "Atomy Psyllium Husk",
        "Atomy Rhodiola Milk Thistle",
        "Atomy Vitamin B Complex",
        "Atomy Lutein",
        "Atomy Lactium",
        "Atomy Pu'er Tea Yeşil Çay",
        "Atomy Hemohim",
    ],

    # =========================
    # Doctorem International
    # =========================
    "doctorem": [
        "Doctorem Vita Plus",
        "Doctorem Gin Plus",
        "Doctorem Body Plus",
        "Doctorem Epifiz Plus",
        "Doctorem Omega Plus",
        "Doctorem Man Plus (Erkek)",
        "Doctorem Woman Plus (Kadın)",
        "Doctorem Thin Plus Belly",
        "Doctorem Thin Plus Normal",
    ],

    # =========================
    # Algophyco / Algo TTS
    # =========================
    "algo_tts": [
        "AlgoTTS POWER PATCH",
        "AlgoTTS OMEGA & KOENZİM PATCH",
        "AlgoTTS SMART PATCH",
        "AlgoTTS PİNEAL PATCH",
        "AlgoTTS MAN PATCH (Erkek)",
        "AlgoTTS WOMAN PATCH (Kadın)",
        "AlgoTTS QUEEN ROYAL JELLY",
        "AlgoTTS FIT PATCH",
        "AlgoTTS BITTER MELON",
        "AlgoTTS STARMOON STRONG COFFEE",
        "AlgoTTS LIPOVIT",
        "AlgoTTS XXL ALGO GINSENG 365",
        "AlgoTTS VIT B PRO",
        "AlgoTTS OUTOVİT",
        "AlgoTTS ALGOFİT",
        "AlgoTTS NEUROVİT",
        "AlgoTTS ARGININE PLUS",
        "AlgoTTS ALGO MEGA",
        "AlgoTTS VİTOMA",
        "AlgoTTS PROTAMİN",
        "AlgoTTS RADİX PRO",
    ],
    # Eski kayıtlar için geriye dönük uyumluluk
    "algo": [
        "AlgoTTS POWER PATCH",
        "AlgoTTS OMEGA & KOENZİM PATCH",
        "AlgoTTS SMART PATCH",
        "AlgoTTS PİNEAL PATCH",
        "AlgoTTS MAN PATCH (Erkek)",
        "AlgoTTS WOMAN PATCH (Kadın)",
        "AlgoTTS QUEEN ROYAL JELLY",
        "AlgoTTS FIT PATCH",
        "AlgoTTS BITTER MELON",
        "AlgoTTS STARMOON STRONG COFFEE",
        "AlgoTTS LIPOVIT",
        "AlgoTTS XXL ALGO GINSENG 365",
        "AlgoTTS VIT B PRO",
        "AlgoTTS OUTOVİT",
        "AlgoTTS ALGOFİT",
        "AlgoTTS NEUROVİT",
        "AlgoTTS ARGININE PLUS",
        "AlgoTTS ALGO MEGA",
        "AlgoTTS VİTOMA",
        "AlgoTTS PROTAMİN",
        "AlgoTTS RADİX PRO",
    ],

    # =========================
    # Herbalife  (sadece iç takviye / içecekler)
    # =========================
    "herbalife": [
        "Formül 1 Shake",
        "Herbal Aloe Konsantre İçecek",
        "Formül 1 Çorba",
        "SKIN Collagen Drink Powder",
        "Bitkisel Çay Tozu",
        "Heartwell™",
        "Tri Blend Select",
        "Protein Bar",
        "LiftOff®",
        "Niteworks®",
        "Protein Cips",
        "Formül 3 Pro-Boost",
        "Multi-fiber",
        "Herbalifeline® Max",
        "Xtra-Cal®",
        "Pro-Drink",
        "Vitamin-Mineral Kadınlar İçin",
        "Vitamin-Mineral Erkekler İçin",
        "Soğuk Kahve Karışımı",
        "Pro-core",
        "Thermo Complete®",
        "Herbalife24® CR7 Drive",
        "Herbalife24® RB ProMax",
    ],

    # =========================
    # Welltures Global
    # =========================
    "welltures": [
        "welltures Gluwell",
        "welltures Omiwell",
        "welltures Multiwell",
        "welltures Suprawell",
        "welltures Epiwell",
        "welltures Bodywell",
        "welltures Admiwell",
        "welltures Maxiwell",
        "welltures Migwell",
        "welltures Frekanswell",
        "welltures Miraclewell",
        "welltures Collagen Face",
        "welltures Collagen Eye",
        "welltures Vitamin C Serum",
        "welltures Hyaluronic Acid Serum",
        "welltures Collagen Serum",
        "welltures Anti Aging Serum",
        "welltures Yüz Temizleme Köpüğü",
        "welltures Sun Stick",
        "welltures Cafewell",
    ],
    "welltures_global": [
        "welltures Gluwell",
        "welltures Omiwell",
        "welltures Multiwell",
        "welltures Suprawell",
        "welltures Epiwell",
        "welltures Bodywell",
        "welltures Admiwell",
        "welltures Maxiwell",
        "welltures Migwell",
        "welltures Frekanswell",
        "welltures Miraclewell",
        "welltures Collagen Face",
        "welltures Collagen Eye",
        "welltures Vitamin C Serum",
        "welltures Hyaluronic Acid Serum",
        "welltures Collagen Serum",
        "welltures Anti Aging Serum",
        "welltures Yüz Temizleme Köpüğü",
        "welltures Sun Stick",
        "welltures Cafewell",
    ],

    # =========================
    # Now International
    # =========================
    "now": [
        "Pain End",
        "Hex Now",
        "Mig Ver",
        "Vit Now",
        "Sleep",
        "TEAMAZING",
        "REDUCE",
        "PRO TEA",
        "Fit Now",
        "Energy",
        "STAY UP",
        "Vit D3 K2",
        "Now Classic Reishi Mantar",
        "CBD Oil",
        "NOW PLUS 5.0 CBD Oil",
        "Hermona Plus Oral Sprey",
    ],

    # =========================
    # Forever Living
    # (çoğunlukla gıda takviyesi ve fonksiyonel içecekler)
    # =========================
    "forever": [
        "Eco 9 Vanilla",
        "Eco 9 Chocolate",
        "C9 Forever Lite Ultr Chocolate Pouch",
        "C9 Forever Lite Ultr Vanilla Pouch",
        "F15 Beginner Ultr Chocolate Pouch",
        "Clean C9 - Forever Lite Ultr Vanilla",
        "Clean C9 - Forever Lite Ultr Chocolate",
        "Start your Journey pack Chocalate",
        "My Fit 1",
        "My Fit 2",
        "My Fit 3",
        "My Fit 4",
        "My Fit 11",
        "My Fit 6",
        "Aloe Vera Gel",
        "Aloe Berry Nectar",
        "Aloe Vera Gelly",
        "Forever Bee Pollen",
        "Forever Bee Propolis",
        "Forever Royal Jelly",
        "Nature Min",
        "Arctic-Sea Omega-3",
        "Absorbent-C",
        "Forever Freedom",
        "Aloe Blossom Herbal Tea",
        "Forever Calcium",
        "Forever Active HA",
        "Forever NutraQ10",
        "Forever Immublend",
        "Vitolize Women’s",
        "Forever Daily",
        "Forever Therm",
        "Forever Fiber",
        "Forever ARGI+ - Pouch",
        "Forever Immune Gummy",
        "Forever Active PRO B",
        "Forever Marine Collagen",
        "Forever Plant Protein",
        "Forever Sensatiable",
        "Forever Absorbent-D",
        "Forever Aloe Mango",
    ],

    # =========================
    # Siberian Wellness
    # (bilinçli filtre: sadece gıda takviyeleri & vitaminler)
    # =========================
    "siberian": [
        # Essential Botanics
        "Siberian Wellness Essential Botanics Aronia & Lutein",
        "Siberian Wellness Essential Botanics Bearberry & Lingonberry",
        # Essential Minerals
        "Essential Minerals IODINE",
        "Essential Minerals IRON",
        "Essential Minerals MAGNESIUM",
        "Essential Minerals ORGANIC ZINC",
        "Essential Minerals ORGANIK KALSIYUM",
        "Essential Minerals SELENIUM",
        # Essential Sorbents / Lymphosan
        "Siberian Wellness Essential Sorbents JOINT COMFORT",
        "Siberian Wellness INULIN CONCENTRATE",
        # Essential Vitamins
        "Alfa lipoik asit",
        "C vitamini ve rutin",
        "Siberian Wellness Essential Vitamins B-COMPLEX & BETAINE",
        "Siberian Wellness Essential Vitamins BEAUTY VITALS",
        "Siberian Wellness Essential Vitamins VITAMIN D3",
        # Novomin
        "Siberian Wellness N.V.M.N. FORMULA 4",
        # Diğer kompleksler
        "Siberian Wellness RENAISSANCE TRIPLE SET",
        "Siberian Health SYNCHROVITALS II",
        "Siberian Health SYNCHROVITALS IV",
        "Siberian Wellness SYNCHROVITALS V",
        "Beta-Carotene in Sea-Buckthorn Oil",
        "Siberian Wellness Trimegavitals LUTEIN AND ZEAKSANTIN SUPERCONCANTRATE",
        # Vitamama serisi
        "Dino Vitamino Syrup with Vitamins and Minerals",
        "VITAMAMA Immunotops Syrup",
        # Women's Health
        "D-mannose & Cranberry extract",
        "Hyaluronic Acid & Natural Vitamin C",
        "Хронолонг (Chronolong)",
        # Beslenme
        "Young & Beauty",
    ],

    # =========================
    # Amare Global
    # =========================
    "amare": [
        "Amare Sunrise",
        "Amare Sunset",
        "Amare Nitro Plus",
        "HL5 Kolajen Protein",
        "FIT20 Whey Protein",
        "RESTORE",
        "ON SHOTS",
        "ORIGIN",
        "R-STOʊR",
        "EDGE",
        "NRGI",
        "MNTA",
        "IGNT HER",
        "IGNT HIM",
        "Wellness Üçgeni Seti",
    ],

    # =========================
    # Onyedi Wellness
    # =========================
    "onyedi": [
        "Onyedi Kolajen",
        "Onyedi K-Patch Ginseng Ağrı Bandı",
        "Onyedi Classic Coffee Reishi Kahve",
        "Onyedi Mocha Coffee Reishi Kahve",
        "Onyedi Latte Coffee Reishi Latte",
        "Onyedi Hot Chocolate Coffee Reishi Sıcak Çikolata",
    ],

    # =========================
    # Oxo Global
    # (ağırlık takviye & iç destek + transdermal)
    # =========================
    "oxo": [
        "LaGrâce Multi Collagen",
        "Booster Shot",
        "Booster Patch",
        "Immunplus",
        "Nutrimeal Live Fit",
        "Melatonin Sleep Patch",
        "Multicollagen 10.000 mg Şase",
        "POWER & PAIN PATCH",
        "Gusto Classico Coffee Reishi mantar ekstraktı",
    ],

    # =========================
    # Som International
    # (takviyeler & iç destek)
    # =========================
    "som": [
        "Som Power",
        "Som King",
        "Som Queen",
        "Som Kids",
        "Som Force",
        "Best Fell",
        "7 Plus 21 Detox",
        "Som Mag Plus",
        "Ionic Water",
        "Slim Form Tea",
        "Active Pro",
        "Som Mist",
        "Som Coffee",
        "Power Focus Coffee",
    ],

    # =========================
    # İndeva Global
    # (özellikle transdermal patch + temel Mary bakım ürünleri)
    # =========================
    "indeva": [
        "Bye Pain",
        "Bye Suger",
        "Bye Stress",
        "Bye Lack",
        "Bye Nox",
        "Mary Saç Bakım Şampuanı",
        "Mary Saç Bakım Serumu",
        "Mary Sakal Bakım Serumu",
        "Mary Dry Oil",
        "Mary Hyaluronik Asit Duş Jeli",
        "Mary Tea Tree Duş Jeli",
        "Mary Nemlendirici Duş Jeli (Çilek & Hindistan Cevizi)",
        "Mary Nemlendirici Duş Jeli (Ahududu & Meyve Özleri)",
        "Mary Ferahlatıcı Erkek Duş Jeli (Bergamot & Yasemin)",
        "Mary Body Scrub",
        "Mary Prufresh Yüz Yıkama Jeli",
        "Mary Revitaluxe Anti Aging Serum",
        "Mary Liftension Collagen Serum",
        "Mary Liftension Collagen Complex Krem",
        "Mary Hydro 3D Krem",
    ],
}


def normalize_complaint_brand(brand: str) -> str:
    """
    Marka adını normalize edip COMPLAINT_BRAND_PRODUCTS içinde
    kullanılabilir bir anahtar haline getirir.
    """
    if not brand:
        return "onemore"

    b = str(brand).lower().strip()

    # Doğrudan eşleşiyorsa olduğu gibi kullan
    if b in COMPLAINT_BRAND_PRODUCTS:
        return b

    # Welltures varyasyonları
    if b in ("welltures global", "welltures_global", "welltures-global"):
        return "welltures_global"

    # Diğer durumlarda da en azından normalize geri dön
    return b

def infer_gender_from_text(text: str) -> str:
    """
    Şikâyet metninden veya özetten yaklaşık cinsiyet tahmini yapar.
    'male' / 'female' / '' döner.
    """
    if not text:
        return ""

    t = str(text).lower()

    # Kadın için ipuçları
    female_tokens = [
        "kadın", "kadin", "bayan", "hanım", "hanim",
        "kız", "kiz", "abla", "hanımefendi", "mrs", "ms "
    ]
    if any(tok in t for tok in female_tokens):
        return "female"

    # Erkek için ipuçları
    male_tokens = [
        "erkek", "bay", "bey", "beye", "beyi",
        "beyefendi", "mr "
    ]
    if any(tok in t for tok in male_tokens):
        return "male"

    return ""


def apply_brand_products_to_complaint(
    analysis: dict,
    brand: str,
    gender: str = "",
) -> dict:
    """
    Şikâyet analizinden gelen 'onemore_urun_onerileri' listesini,
    seçilen markanın KENDİ ürün isimleriyle hizalar.

    - Ürün isimlerini get_brand_products(brand) fonksiyonundan alır
    - Model yanlış marka yazsa bile burada düzeltilir
    - 'neden' ve 'sure' metinleri mümkün olduğunca korunur
    - gender = 'male' / 'female' ise karşı cins ürünleri ELER
    """
    if not isinstance(analysis, dict):
        return analysis

    # ------------------------------------------------
    # CİNSİYETE GÖRE UYGUN / UYGUN DEĞİL ÜRÜN KONTROLÜ
    # ------------------------------------------------
    def is_gender_compatible(product_name: str, gender_val: str) -> bool:
        """
        Ürün adında 'Women, Kadın, Woman, Lady, Female' vb. varsa
        erkek için önermeyelim; tam tersi de geçerli.
        """
        name = (product_name or "").lower()
        if not gender_val:
            return True

        female_words = [
            "women", "woman", "female", "kadın", "kadin", "bayan",
            "vitolize women", "woman patch", "kadın patch"
        ]
        male_words = [
            "men", "man", "male", "erkek", "bay", "bey",
            "man patch", "erkek patch"
        ]

        if gender_val == "male":
            # Erkek için kadın ürünlerini ele
            if any(w in name for w in female_words):
                return False
        elif gender_val == "female":
            # Kadın için erkek ürünlerini ele
            if any(w in name for w in male_words):
                return False

        return True

    # Modelden gelen liste (varsa)
    raw_list = analysis.get("onemore_urun_onerileri")
    if not isinstance(raw_list, list):
        raw_list = []

    # Markaya göre KESİN ürün listesi
    try:
        canon_products = get_brand_products(brand) or []
    except Exception:
        return analysis

    if not canon_products:
        return analysis

    # Cinsiyet clean
    gender = (gender or "").strip().lower()
    if gender not in ("male", "female"):
        gender = ""

    # İlk hedef uzunluğu belirle (en az 3 ürün gösterelim)
    if raw_list:
        target_len = min(len(raw_list), len(canon_products))
    else:
        target_len = min(3, len(canon_products))

    new_items = []
    raw_idx = 0

    for p in canon_products:
        if len(new_items) >= target_len:
            break

        # Cinsiyete uygun değilse atla
        if not is_gender_compatible(p, gender):
            continue

        base = {}
        if raw_idx < len(raw_list) and isinstance(raw_list[raw_idx], dict):
            base = raw_list[raw_idx]
            raw_idx += 1

        reason = base.get("neden") or base.get("aciklama") or ""
        duration = base.get("sure") or "8 hafta"

        new_items.append({
            "urun": p,
            "neden": reason or "Şikâyet setine göre uygun destekleyici bir wellness ürünüdür.",
            "sure": duration,
        })

    # Hiç ürün kalmadıysa (çok agresif filtre olduysa), cinsiyet filtresini gevşetip tekrar dolduralım
    if not new_items:
        for p in canon_products[:target_len]:
            base = {}
            if raw_idx < len(raw_list) and isinstance(raw_list[raw_idx], dict):
                base = raw_list[raw_idx]
                raw_idx += 1

            reason = base.get("neden") or base.get("aciklama") or ""
            duration = base.get("sure") or "8 hafta"

            new_items.append({
                "urun": p,
                "neden": reason or "Şikâyet setine göre uygun destekleyici bir wellness ürünüdür.",
                "sure": duration,
            })

    analysis["onemore_urun_onerileri"] = new_items
    return analysis




def build_complaint_prompt(
    complaint_text: str,
    target_lang: str,
    second_lang: str,
    brand: str,
) -> str:
    brand_label = BRAND_LABELS.get(brand, "OneMore International")
    products = get_brand_products(brand)
    products_block = "\n".join(f"- {p}" for p in products)
    lang_instruction = build_language_instruction(target_lang, second_lang, source="complaint")


# TIBBİ SORUMLULUK BEYANI İÇİN KURAL
# - Tıbbi sorumluluk beyanında ASLA marka adı, cihaz adı kullanma.
# - "Bu rapor" veya "bu değerlendirme" gibi nötr ifadeler kullan.
# - OneMore, Xaura, WellnessAnalyzer, cihaz adı vb. MARKA VE CİHAZ isimleri YAZMA.


    prompt = f"""
Aşağıda bir kişinin anlattığı şikâyet ve mevcut durumu var.
Bu metne göre ŞİKÂYET BAZLI bir WELLNESS RAPORU hazırla.

ÇALIŞILAN MARKA:
- Marka adı: {brand_label}
- Ürün önerilerini SADECE bu markanın ürünlerinden seç.
- FARKLI MARKALARA AİT ÜRÜN İSİMLERİNİ ASLA KULLANMA.
- Aşağıdaki ürün listesi dışına ÇIKMA, yeni ürün ismi uydurma.

ÖZEL DURUMLAR:
- Eğer marka 'Genel wellness analizi (markasız)' ise:
  marka ismi kullanma, sadece ürün/etki TİPİ yaz.
- Eğer marka OneMore dışındaki bir marka ise:
  OneMore ürünlerini ASLA yazma.

DİL KURALI:
{lang_instruction}

ŞİKÂYET METNİ:
------------------
{complaint_text}
------------------

ÜRÜN KURALI:
- 'onemore_urun_onerileri' listesinde EN AZ 6, EN FAZLA 8 ürün olsun.
- Her üründe:
  - 'urun': Aşağıdaki ürün listesinden birebir isim.
  - 'neden': Şikâyet setine göre en az 2 cümlelik net açıklama.
  - 'sure': Örn. '4 hafta', '8 hafta'.

Tıbbi teşhis koyma, ilaç ismi verme. Sadece sağlıklı yaşam,
takviye ve {brand_label} ürün önerileri ile sınırlı kal.
Risk değerlendirmesi ve yaşam önerileri bölümlerinde
GENİŞ ve AÇIKLAYICI ol (toplamda en az 25–40 cümle).

ÇIKTIYI SADECE GEÇERLİ BİR JSON OLARAK DÖN.

JSON şeman:

{{
  "sikayet_ozeti": "",
  "olasi_sistem_yukleri": [
    {{
      "sistem_adi": "",
      "gerekce": ""
    }}
  ],
  "risk_degerlendirmesi": "",
  "yasam_onerileri": "",
  "onemore_urun_onerileri": [
    {{
      "urun": "",
      "neden": "",
      "sure": ""
    }}
  ],
  "tibbi_sorumluluk": ""
}}

{brand_label} ürünlerini yalnızca şu listeden seç:

{products_block}

SADECE GEÇERLİ JSON DÖN; açıklama yazma, Markdown veya ``` kullanma.
"""
    return prompt


def build_compare_prompt(
    old_text: str,
    new_text: str,
    target_lang: str,
    second_lang: str,
    brand: str,
) -> str:
    brand_label = BRAND_LABELS.get(brand, "OneMore International")
    products = get_brand_products(brand)
    products_block = "\n".join(f"- {p}" for p in products)
    lang_instruction = build_language_instruction(target_lang, second_lang, source="compare")

    # TIBBİ SORUMLULUK BEYANI İÇİN KURAL
    # - Tıbbi sorumluluk beyanında ASLA marka adı, cihaz adı kullanma.
    # - "Bu rapor" veya "bu değerlendirme" gibi nötr ifadeler kullan.
    # - OneMore, Xaura, WellnessAnalyzer, cihaz adı vb. MARKA VE CİHAZ isimleri YAZMA.

    prompt = f"""
Aşağıda aynı kişiye ait iki farklı Wellness Analyzer TEST PDF'inin metni var.
Biri ESKİ TEST, diğeri YENİ TEST. Bu iki testi karşılaştır.

ÇALIŞILAN MARKA:
- Marka adı: {brand_label}
- Ürün önerilerini SADECE bu markanın ürünlerinden seç.
- FARKLI MARKALARA AİT ÜRÜN İSİMLERİNİ ASLA KULLANMA.
- Aşağıdaki ürün listesi dışına ÇIKMA, yeni ürün ismi uydurma.

ÖZEL DURUMLAR:
- Eğer marka 'Genel wellness analizi (markasız)' ise:
  - Marka ismi kullanma, sadece ürün/etki TİPİ veya KATEGORİ yaz.
  - Örnek: "omega-3 takviyesi", "probiyotik takviyesi", "bitkisel detoks çayı",
    "multivitamin + mineral kompleksi", "C vitamini takviyesi", "magnezyum + B6 takviyesi".
- Eğer marka OneMore dışındaki bir marka ise:
  OneMore ürünlerini ASLA yazma.

DİL KURALI:
{lang_instruction}

ESKİ TEST METNİ:
------------------
{old_text}
------------------

YENİ TEST METNİ:
------------------
{new_text}
------------------

ÜRÜN KURALI:
- 'onemore_urun_onerileri' listesinde EN AZ 6, EN FAZLA 8 ürün olsun.
- Her üründe:
  - 'urun':
      * Eğer marka 'Genel wellness analizi (markasız)' ise: ürün TİPİ/KATEGORİ yaz
        (örneğin "omega-3 takviyesi", "probiyotik takviyesi", "koenzim q10", "C vitamini takviyesi").
      * Diğer markalarda: aşağıdaki ürün listesinden birebir ürün ismi yaz.
  - 'neden': Eski ve yeni test arasındaki DEĞİŞİME göre en az 2 cümlelik açıklama.
  - 'sure': Örn. '4 hafta', '8 hafta'.

4 HAFTALIK TAKİP PLANI KURALI:
- 'dort_haftalik_plan' ALANI MUTLAKA 4 ELEMANLI BİR LİSTE OLACAKTIR.
- Her elemanda şu alanlar ZORUNLUDUR:
  'hafta', 'odak', 'detay', 'urun_kullanimi'.
- 'hafta' alanları SIRAYLA 1, 2, 3 ve 4 olmalıdır.
- Her hafta için:
  - 'odak': O haftanın ana odağını kısaca özetle (örneğin:
    "kardiyovasküler dengeleme", "sindirim ve detoks", "bağışıklık güçlendirme",
    "uyku ve hormonal denge").
  - 'detay': Eski ve yeni test sonuçlarındaki değişimi dikkate alarak,
    o hafta yapılacak yaşam tarzı değişikliklerini (beslenme, uyku, stres, hareket vb.)
    açıklayan en fazla 2 cümle yaz.
  - 'urun_kullanimi': O haftada kullanılacak ürün veya ürün tiplerini ve sürelerini
    kısaca yaz (örnek: "omega-3 takviyesi ve probiyotik; her gün, 4 hafta").
- Kesinlikle sadece 1. haftayı yazıp bırakma; 2., 3. ve 4. haftaları da DOLDUR.

ÇIKTIYI SADECE GEÇERLİ JSON OLARAK DÖN.

JSON şeman:

{{
  "kisi_bilgileri": {{
    "ad_soyad": "",
    "yas": "",
    "cinsiyet": ""
  }},
  "genel_degerlendirme": "",
  "vucut_formu_karsilastirma": "",
  "iyilesen_sistemler": [
    {{
      "sistem_adi": "",
      "degisim": ""
    }}
  ],
  "kotulesen_sistemler": [
    {{
      "sistem_adi": "",
      "degisim": ""
    }}
  ],
  "stabil_sistemler": [
    {{
      "sistem_adi": "",
      "degisim": ""
    }}
  ],
  "dort_haftalik_plan": [
    {{
      "hafta": 1,
      "odak": "",
      "detay": "",
      "urun_kullanimi": ""
    }},
    {{
      "hafta": 2,
      "odak": "",
      "detay": "",
      "urun_kullanimi": ""
    }},
    {{
      "hafta": 3,
      "odak": "",
      "detay": "",
      "urun_kullanimi": ""
    }},
    {{
      "hafta": 4,
      "odak": "",
      "detay": "",
      "urun_kullanimi": ""
    }}
  ],
  "onemore_urun_onerileri": [
    {{
      "urun": "",
      "neden": "",
      "sure": ""
    }}
  ],
  "tibbi_sorumluluk": ""
}}

{brand_label} ürünlerini yalnızca şu listeden seç:

{products_block}

SADECE GEÇERLİ JSON DÖN; açıklama yazma, Markdown veya ``` kullanma.
"""
    return prompt


# ====================================================
#  OPENAI ÇAĞRISI  (CHAT COMPLETIONS + JSON TAMİR)
# ====================================================

async def repair_json_with_openai(broken_json_str: str) -> dict:
    """
    Modelden gelen BOZUK / YARIM JSON'u,
    ikinci bir OpenAI çağrısıyla tamir eder.
    Şema mümkün olduğunca korunur, metinler gerekirse KISALTILIR.
    """
    fix_prompt = f"""
Sen bir JSON DÜZELTME motorusun.

Aşağıda BOZUK veya YARIM KALMIŞ bir JSON metni var.

GÖREVİN:
- Aynı ana şemayı ve alan isimlerini mümkün olduğunca KORU.
- Liste ve sözlük (object) yapısını korumaya çalış.
- HATALI, YARIM kalmış cümleleri kısaltabilir veya tamamen silebilirsin.
- GEREKSİZ UZUN açıklamalar ekleme. Mevcut metni toparlayıp kısaltabilirsin.
- ÇIKTIN TAMAMEN GEÇERLİ JSON olsun.
- JSON DIŞINDA tek bir kelime bile yazma (Markdown, açıklama vs. YASAK).

BOZUK_JSON:
{broken_json_str}
"""

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=fix_prompt,
        )

        first_item = resp.output[0].content[0]
        repaired_content = getattr(first_item, "text", None)
        if repaired_content is None:
            try:
                repaired_content = json.dumps(first_item, ensure_ascii=False)
            except Exception:
                repaired_content = str(first_item)

        print("=== JSON REPAIR HAM İÇERİK ===")
        print("Ham içerik (repr):", repr(repaired_content))

        return json.loads(repaired_content)

    except Exception as e:
        print("=== JSON REPAIR HATASI ===")
        print(e)
        raise ValueError(f"Bozuk JSON'u tamir ederken hata oluştu: {e}") from e


import json
from typing import Any, Dict

# ============================================================
#   JSON MODE PARSE + REPAIR FONKSİYONU (YENİ)
# ============================================================
def parse_json_mode_payload_with_repair(raw: str) -> Dict[str, Any]:
    """
    OpenAI JSON MODE çıktısını güvenli şekilde parse eder:
    - Kod bloklarını (```json ... ```) temizler
    - İlk "{" ile son "}" arasını keser
    - json.loads ile tek seferde parse etmeyi dener
    """

    print("=== JSON MODE HAM İÇERİK ===")
    print(f"Ham içerik (repr): {raw!r}")

    if not raw:
        raise ValueError("Boş JSON içeriği")

    text = str(raw).strip()

    # 1) Eğer ```json blokluysa temizle
    if text.startswith("```"):
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1:
            text = text[first_brace:last_brace + 1].strip()

    # 2) Yine de ilk { - son } arası garanti olsun
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1:
        candidate = text[first_brace:last_brace + 1].strip()
    else:
        candidate = text

    # 3) JSON parse
    try:
        return json.loads(candidate)
    except Exception as e:
        print("=== JSON PARSE HATASI ===")
        print(e)
        raise ValueError("JSON parse edilemedi — içerik tamamen bozuk görünüyor.")

def _clean_json_from_model(content: str) -> str:
    """
    Modelden gelen metni temizler:
    - ```json / ``` kod bloklarını atar
    - İlk '{' ile son '}' arasını alır
    """
    if not isinstance(content, str):
        content = str(content)

    text = content.strip()

    # 1) Başta ``` veya ```json varsa, satır satır temizle
    if text.startswith("```"):
        lines = text.splitlines()
        # İlk satır ``` veya ```json -> at
        if lines:
            lines = lines[1:]
        # Son satır da ``` ise onu da at
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 2) İlk '{' ile son '}' arasını al
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first:last + 1].strip()

    return text

# ============================================================
#   OpenAI CALL (TAM GÜNCEL VE HATASIZ)
# ============================================================
async def call_openai_with_prompt(prompt: str, max_tokens: int = 6500) -> dict:
    """
    OpenAI'yi mevcut SDK sürümüne uygun şekilde çağırır.
    - response_format KULLANMIYORUZ.
    - Çıktıyı text olarak alıp json.loads ile parse ediyoruz.
    - Kod bloklarındaki ```json ... ``` kısımlarını temizliyoruz.
    """

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )

        first_item = resp.output[0].content[0]
        content = getattr(first_item, "text", None)
        if content is None:
            try:
                content = json.dumps(first_item, ensure_ascii=False)
            except Exception:
                content = str(first_item)

    except Exception as e:
        print("=== OPENAI İSTEK HATASI ===")
        print(e)
        raise RuntimeError(f"OpenAI isteği sırasında hata oluştu: {e}") from e

    # 1) Ham içerik (debug)
    print("=== JSON MODE HAM İÇERİK ===")
    print("Ham içerik (repr):", repr(content))

    # 2) Kod bloğu ve gereksiz kısımları temizle
    cleaned = _clean_json_from_model(content)

    print("=== TEMİZLENMİŞ JSON METNİ ===")
    print("Temiz (repr):", repr(cleaned))

    # 3) JSON parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("=== JSON PARSE HATASI ===")
        print("Hata:", e)
        raise RuntimeError(f"OpenAI'den gelen JSON çözümlenemedi: {e}") from e


# ====================================================
#  46 SİSTEM KARTINI ZORUNLU HALE GETİREN YAPI
# ====================================================

def normalize_system_title_for_match(title: str) -> str:
    """
    Sistem adını kıyaslama için normalize eder:
    - Baştaki "11. " gibi numaraları temizler
    - Küçük harfe çevirir
    - Fazla boşlukları siler
    """
    s = str(title or "")
    s = re.sub(r"^\d+\.\s*", "", s)  # baştaki "11. " vb. temizle
    s = s.strip().lower()
    return s


def ensure_46_system_cards(analysis: dict) -> dict:
    """
    analysis["sistem_kartlari"] içinde HER ZAMAN 46 kart olmasını sağlar ama:
    - Modelin ürettiği kartların içeriğini BOZMAYIZ.
    - Eğer zaten 40 veya daha fazla kart varsa, HİÇ DOKUNMAYIZ (sırayı ve metni koruruz).
    - Daha az kart varsa, eksik sistemler için SONA boş kartlar ekleriz.
    """
    if not isinstance(analysis, dict):
        return analysis

    existing_cards = analysis.get("sistem_kartlari") or []
    if not isinstance(existing_cards, list):
        existing_cards = []

    # Model zaten 40+ kart ürettiyse, ona güveniyoruz, hiçbir şeye dokunmuyoruz.
    if len(existing_cards) >= 40:
        analysis["sistem_kartlari"] = existing_cards
        return analysis

    # Var olan kartların norm adlarını topla
    used_norms = set()
    for card in existing_cards:
        if not isinstance(card, dict):
            continue
        raw_name = card.get("sistem_adi") or ""
        norm = normalize_system_title_for_match(raw_name)
        if norm:
            used_norms.add(norm)

    # Eksik olan sistemler için SONDA boş kart ekle
    for full_name in SYSTEM_NAMES:
        clean_title = re.sub(r"^\d+\.\s*", "", full_name).strip()
        norm = normalize_system_title_for_match(clean_title)

        if norm in used_norms:
            continue

        empty_card = {
            "sistem_adi": clean_title,
            "durum": "",
            "belirtiler": "",
            "riskler": "",
            "yasam_tavsiyesi": "",
            "urun_onerileri": [],
        }
        existing_cards.append(empty_card)
        used_norms.add(norm)

        if len(existing_cards) >= 46:
            break

    analysis["sistem_kartlari"] = existing_cards
    return analysis


def _filter_product_list(
    raw_list: List[Dict[str, Any]],
    allowed_products_lower: set,
    brand: str,
) -> List[Dict[str, Any]]:
    """
    Bir ürün listesini (onemore_urun_onerileri / urun_onerileri) marka filtresinden geçirir.
    """
    if not isinstance(raw_list, list):
        return []

    result = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue

        name = str(
            item.get("urun")
            or item.get("urun_adi")
            or item.get("name")
            or item.get("title")
            or ""
        ).strip()

        if not name:
            continue

        name_lower = name.lower()

        if brand not in ("generic", "pati"):
            if name_lower not in allowed_products_lower:
                continue

        if brand in ("generic", "pati"):
            for bkey, lst in PRODUCT_LISTS.items():
                for pname in lst:
                    if pname.lower() in name_lower:
                        name = ""
                        break

        if not name and brand in ("generic", "pati"):
            continue

        cleaned = {
            "urun": name,
            "neden": item.get("neden", "") or "",
            "sure": item.get("sure", "") or "",
        }
        result.append(cleaned)

    return result


def apply_brand_product_filter(analysis: dict, brand: str) -> dict:
    """
    Analiz içindeki ürün önerilerini seçilen markaya göre filtreler.
    - Marka ürün listesinde olmayan ürün isimlerini atar (markalı analizlerde).
    - 'generic' (markasız) analizlerde ürün önerilerine DOKUNMAZ,
      sadece kart sayısını tamamlar.
    - 'pati' markasında ve generic'te sert filtre uygulamaz, isim temizliği
      _filter_product_list içindeki mantığa göre yapılır.
    - 46 sistem kartının sayısını bozmaz.
    """
    if not isinstance(analysis, dict):
        return analysis

    # Her durumda önce kart sayısını garanti altına al
    analysis = ensure_46_system_cards(analysis)

    # Markasız (generic) analizde ürün isimlerine dokunmuyoruz;
    # LLM'nin ürettiği vitamin-mineral-bitki önerileri kalsın.
    if brand == "generic":
        return analysis

    # Diğer markalar için izin verilen ürünleri hazırla
    allowed = {p.lower() for p in get_brand_products(brand)}

    # Genel yardımcı
    def _fix_list(raw):
        return _filter_product_list(raw, allowed, brand)

    # GENEL BULGU
    gb = analysis.get("genel_bulgu")
    if isinstance(gb, dict):
        # Eski alan (onemore_urun_onerileri) korunuyor
        gb["onemore_urun_onerileri"] = _fix_list(
            gb.get("onemore_urun_onerileri", [])
        )
        # Yeni, markadan bağımsız alan
        gb["urun_onerileri"] = _fix_list(
            gb.get("urun_onerileri", [])
        )

    # KANALLAR & KOLLATERALLER
    kk = analysis.get("kanallar_ve_kollateraller_detay")
    if isinstance(kk, dict):
        kk["onemore_urun_onerileri"] = _fix_list(
            kk.get("onemore_urun_onerileri", [])
        )
        kk["urun_onerileri"] = _fix_list(
            kk.get("urun_onerileri", [])
        )

    # İNSAN BİLİNÇ DÜZEYİ
    ib = analysis.get("insan_bilinc_duzeyi_detay")
    if isinstance(ib, dict):
        ib["onemore_urun_onerileri"] = _fix_list(
            ib.get("onemore_urun_onerileri", [])
        )
        ib["urun_onerileri"] = _fix_list(
            ib.get("urun_onerileri", [])
        )

    # 46 SİSTEM KARTI
    cards = analysis.get("sistem_kartlari")
    if isinstance(cards, list):
        for kart in cards:
            if not isinstance(kart, dict):
                continue
            kart["urun_onerileri"] = _fix_list(
                kart.get("urun_onerileri", [])
            )

    # ANALYSIS GENEL ÜRÜN ÖNERİLERİ (varsa)
    if "onemore_urun_onerileri" in analysis:
        analysis["onemore_urun_onerileri"] = _fix_list(
            analysis.get("onemore_urun_onerileri", [])
        )
    if "urun_onerileri" in analysis:
        analysis["urun_onerileri"] = _fix_list(
            analysis.get("urun_onerileri", [])
        )

    return analysis


def apply_gender_card_filter(analysis: dict) -> dict:
    if not isinstance(analysis, dict):
        return analysis

    kisi = analysis.get("kisi_bilgileri", {}) or {}
    cinsiyet_raw = str(kisi.get("cinsiyet", "")).strip()
    gender = normalize_gender(cinsiyet_raw)

    if gender not in ("male", "female"):
        return analysis

    blocked_for_male = [
        "Meme (kadın)",
        "Adet döngüsü",
        "Kadınlık Hormonu",
        "Kadın Hormonu",
        "(kadın)",
    ]
    blocked_for_female = [
        "Prostat",
        "Erkek Cinsel Fonksiyonu",
        "Sperm ve meni",
        "Erkek Hormonu",
        "(erkek)",
    ]

    cards = analysis.get("sistem_kartlari")
    if isinstance(cards, list):
        for kart in cards:
            if not isinstance(kart, dict):
                continue
            name = str(kart.get("sistem_adi", "")).strip()
            name_lower = name.lower()

            if gender == "male":
                if any(token.lower() in name_lower for token in blocked_for_male):
                    kart["durum"] = ""
                    kart["belirtiler"] = ""
                    kart["riskler"] = ""
                    kart["yasam_tavsiyesi"] = ""
                    kart["urun_onerileri"] = []
            elif gender == "female":
                if any(token.lower() in name_lower for token in blocked_for_female):
                    kart["durum"] = ""
                    kart["belirtiler"] = ""
                    kart["riskler"] = ""
                    kart["yasam_tavsiyesi"] = ""
                    kart["urun_onerileri"] = []

    return analysis


def fill_empty_system_cards(analysis: dict) -> dict:
    """
    Tamamen boş kartları minimum seviyede doldurur:
    - Erkek için: Prostat / Erkek Cinsel Fonksiyonu / Sperm ve meni / Erkek Hormonu
      kartları boşsa KISA açıklamalar yazar.
    - Kadın için: Meme / Adet döngüsü / Kadınlık hormonu kartları boşsa KISA açıklamalar yazar.
    - Diğer sistemlerde otomatik doldurma yapmaz (boş kalabilir).
    """
    if not isinstance(analysis, dict):
        return analysis

    kisi = analysis.get("kisi_bilgileri") or {}
    cinsiyet_raw = str(kisi.get("cinsiyet", "")).strip()
    gender = normalize_gender(cinsiyet_raw)

    cards = analysis.get("sistem_kartlari")
    if not isinstance(cards, list):
        return analysis

    for kart in cards:
        if not isinstance(kart, dict):
            continue

        name = str(kart.get("sistem_adi", "") or "")
        name_lower = name.lower()

        durum = str(kart.get("durum", "") or "").strip()
        belirtiler = str(kart.get("belirtiler", "") or "").strip()
        riskler = str(kart.get("riskler", "") or "").strip()
        yasam_tavsiyesi = str(kart.get("yasam_tavsiyesi", "") or "").strip()
        urunler = kart.get("urun_onerileri") or []

        has_text = any([durum, belirtiler, riskler, yasam_tavsiyesi])
        has_products = bool(urunler)

        if has_text or has_products:
            continue

        if gender == "male" and any(
            token in name_lower
            for token in ["prostat", "erkek cinsel", "sperm", "meni", "erkek hormonu"]
        ):
            kart["durum"] = (
                "Bu kart, erkek üreme sistemi ve hormon dengesindeki olası yükleri özetler."
            )
            kart["belirtiler"] = (
                "Libido değişiklikleri, enerji dalgalanmaları veya idrar alışkanlıklarında farklılıklar görülebilir."
            )
            kart["riskler"] = (
                "Uzun vadede prostat sağlığı ve hormon dengesi üzerinde olumsuz etkiler ortaya çıkabilir."
            )
            kart["yasam_tavsiyesi"] = (
                "Düzenli üroloji kontrolleri, dengeli beslenme, sigaradan kaçınma ve günlük hareket önemlidir."
            )
            continue

        if gender == "female" and any(
            token in name_lower
            for token in [
                "meme",
                "adet döngüsü",
                "adet dongusu",
                "kadınlık hormonu",
                "kadinlik hormonu",
                "kadın hormonu",
                "kadin hormonu",
            ]
        ):
            kart["durum"] = (
                "Bu kart, kadın üreme sistemi, hormon dengesi ve meme dokusuna ait olası yükleri özetler."
            )
            kart["belirtiler"] = (
                "Adet döngüsünde düzensizlik, göğüs hassasiyeti ve duygu durum dalgalanmaları yaşanabilir."
            )
            kart["riskler"] = (
                "Dengesizlikler uzun vadede adet problemleri ve meme sağlığı üzerinde risk oluşturabilir."
            )
            kart["yasam_tavsiyesi"] = (
                "Düzenli jinekolojik kontroller, rafine şekerden kaçınma, yeterli uyku ve stres yönetimi önerilir."
            )
            continue

        # Diğer kartları boş bırak (otomatik generic cümle basmıyoruz)

    return analysis


# BUNLARI TERCIHEN DOSYANIN EN ÜSTÜNDE TUTABİLİRSİN:
import re
from typing import Optional, Dict, Any, List

CHILD_MAX_AGE = 10  # 10 yaşından KÜÇÜK çocuk olarak kabul edelim

# ÇOCUKTA İÇİ DOLU KALMASINA İZİN VERDİĞİMİZ KARTLAR (anahtar kelime bazlı)
CHILD_ALLOWED_SYSTEM_KEYWORDS = [
    # Element / iz element
    "eser element",
    "iz element",

    # Vitamin – Amino Asit – Koenzim
    "vitamin",
    "amino asit",
    "koenzim",

    # Esansiyel yağ asitleri
    "esansiyel yağ asidi",
    "esansiyel yag asidi",

    # ADHD / dikkat eksikliği
    "adhd",
    "dikkat eksikliği",
    "dikkat eksikligi",

    # Ergen kartları
    "ergen zekası",
    "ergen zekasi",
    "ergen büyüme endeksi",
    "ergen buyume endeksi",

    # Lesitin
    "lesitin",

    # Yağ asidi / yağ asitleri
    "yağ asidi",
    "yağ asit",
    "yag asidi",
    "yag asit",
]


def parse_age_to_int(raw_age):
    """
    PDF / JSON'dan gelen yaş bilgisini güvenli şekilde integer'a çevirir.
    Örnekler:
      "8"       -> 8
      "8 yaş"   -> 8
      "08 "     -> 8
      8         -> 8
    Anlaşılmaz bir şey gelirse None döner ve sistemi bozmamış oluruz.
    """
    if raw_age is None:
        return None

    # Zaten sayı ise
    if isinstance(raw_age, (int, float)):
        try:
            return int(raw_age)
        except Exception:
            return None

    # Metin olarak ele al
    text = str(raw_age).strip()

    # Tüm sayı olmayan karakterleri temizle (yaş, yıl gibi ekleri at)
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None

    try:
        return int(digits)
    except Exception:
        return None

import re  # muhtemelen en üstte zaten var, yoksa ekle

def parse_age_to_int(age_raw):
    """
    PDF analizinden gelen yaş bilgisini güvenli şekilde integer'a çevirir.
    Örnek: "8", "8 yaş", "Yaş: 8", "8.0" -> 8
    """
    if age_raw is None:
        return None

    # Zaten sayı ise
    if isinstance(age_raw, (int, float)):
        try:
            return int(age_raw)
        except Exception:
            return None

    # String ise içinden rakamları çekelim
    if isinstance(age_raw, str):
        # İçinden tüm rakamları yakala
        m = re.search(r"(\d+)", age_raw)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    return None


def is_child_age(age_val: int | None) -> bool:
    """
    ÇOCUK testleri için yaş mantığı:
    - 10 yaş altı (0–9) çocuk kabul edilir.
    - 10 ve üzeri yaşlarda 46 sistemlik yetişkin raporu kullanılır.
    """
    if age_val is None:
        return False
    return age_val < 10


# ============================================
#  ÇOCUK & CİNSİYET KART FİLTRELERİ
# ============================================

CHILD_MAX_AGE = 9  # 0–9 yaş = çocuk, 10 yaş ve üzeri = yetişkin (46 kart)

ADULT_ONLY_CHILD_BLOCK_TOKENS = [
    # Erkek cinsellik / prostat
    "prostat",
    "erkek cinsel",
    "sperm",
    "meni",
    "erkek hormonu",
    "(erkek)",

    # Kadın cinsellik / meme / adet
    "meme",
    "kadınlık hormon",
    "kadinlik hormon",
    "adet",
    "(kadın)",
    "(kadin)",
]

FEMALE_ONLY_TOKENS = [
    "meme",
    "kadın",
    "kadin",
    "adet",
    "kadınlık hormon",
    "kadinlik hormon",
    "(kadın)",
    "(kadin)",
]

MALE_ONLY_TOKENS = [
    "prostat",
    "erkek cinsel",
    "sperm",
    "meni",
    "erkek hormonu",
    "(erkek)",
]


def _extract_age_from_analysis(analysis: dict) -> int | None:
    """
    kisi_bilgileri içinden yaşı güvenli şekilde çek.
    Örnek değerler: 8, "8", "8 yaş", "8 years" vs.
    """
    kisi = analysis.get("kisi_bilgileri") or {}
    yas_raw = kisi.get("yas") or kisi.get("age") or ""

    if yas_raw is None:
        return None

    s = str(yas_raw).strip()
    if not s:
        return None

    # Başındaki rakamları al (örn. "8 yaş" -> "8")
    m = re.match(r"(\d+)", s)
    if not m:
        return None

    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_gender_from_analysis(analysis: dict) -> str:
    """
    cinsiyet bilgisini normalleştir: "male" / "female" / "unisex"
    """
    kisi = analysis.get("kisi_bilgileri") or {}
    g = kisi.get("cinsiyet") or kisi.get("gender") or ""
    return normalize_gender(str(g))


def filter_child_system_cards_if_needed(analysis: dict) -> dict:
    """
    0–9 yaş arası çocuklarda:
      - Prostat, erkek cinsellik, sperm/meni, erkek hormonu
      - Kadın meme, adet, kadınlık hormon kartları
    gibi kartları tamamen listeden kaldırır.
    10 yaş ve üzeri (>=10) için hiçbir şey yapmaz.
    """
    age = _extract_age_from_analysis(analysis)
    if age is None or age > CHILD_MAX_AGE:
        # Yaş yoksa ya da 10+ ise dokunma
        return analysis

    cards = analysis.get("sistem_kartlari")
    if not isinstance(cards, list):
        return analysis

    filtered_cards = []
    for card in cards:
        if not isinstance(card, dict):
            continue

        name = (
            card.get("sistem_adi")
            or card.get("baslik")
            or card.get("title")
            or ""
        )
        name_l = str(name).lower()

        # Çocuklarda kesinlikle istemediğimiz kartlar
        if any(tok in name_l for tok in ADULT_ONLY_CHILD_BLOCK_TOKENS):
            # Tamamen atlıyoruz (listeye eklemiyoruz)
            continue

        filtered_cards.append(card)

    analysis["sistem_kartlari"] = filtered_cards
    return analysis


def apply_gender_card_filter(analysis: dict) -> dict:
    """
    YETİŞKİNLER için cinsiyete ters kartlardaki içeriği temizler.
    (Kartı tamamen silmiyoruz, ama içeriğini 'bu kart değerlendirilmez'
    şeklinde boşaltıyoruz.)
    Çocuk filtresinden SONRA çalışması daha mantıklı.
    """
    gender = _extract_gender_from_analysis(analysis)
    cards = analysis.get("sistem_kartlari")
    if not isinstance(cards, list):
        return analysis

    for card in cards:
        if not isinstance(card, dict):
            continue

        name = (
            card.get("sistem_adi")
            or card.get("baslik")
            or card.get("title")
            or ""
        )
        name_l = str(name).lower()

        # Önce çocuklara özel durum zaten yukarıda çözüldü.
        # Burada yetişkin erkek / kadın ayrımı yapıyoruz.
        if gender == "male":
            # Erkekte kadın spesifik kartları boşalt
            if any(tok in name_l for tok in FEMALE_ONLY_TOKENS):
                card["ozet"] = "Bu kart erkeklerde değerlendirilmez."
                card["detay"] = ""
                card["detay2"] = ""
                card["oneriler"] = ""
                card["urun_onerileri"] = []
        elif gender == "female":
            # Kadında erkek spesifik kartları boşalt
            if any(tok in name_l for tok in MALE_ONLY_TOKENS):
                card["ozet"] = "Bu kart kadınlarda değerlendirilmez."
                card["detay"] = ""
                card["detay2"] = ""
                card["oneriler"] = ""
                card["urun_onerileri"] = []

    return analysis



# ====================================================
#  TEK DİLLİ JSON'U İKİ DİLLİ HALE ÇEVİRME (ŞİMDİLİK PASİF)
# ====================================================

from typing import Any as _Any, List as _List  # sadece tip için


async def make_bilingual_json(analysis: dict, target_lang: str, second_lang: str) -> dict:
    """
    NOT: Yeni yapıda JSON MODE zaten iki dilli (target_lang + second_lang) metin üretiyor.
    Bu yüzden burada ekstra bir OpenAI çağrısı yapmamıza gerek yok.
    Şimdilik analiz JSON'unu olduğu gibi geri döndürüyoruz.
    """
    return analysis


# ====================================================
#  ROUTE'LAR
# ====================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    history = load_history()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "history": history},
    )


@app.get("/report/{report_id}", response_class=HTMLResponse)
async def get_report(report_id: str, request: Request):
    entry = get_history_entry(report_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Rapor bulunamadı.")

    ctx = entry.get("ctx", {}) or {}
    if not isinstance(ctx, dict):
        raise HTTPException(status_code=500, detail="Geçersiz rapor kaydı.")

    # Hangi template ile kaydedilmişse onu kullan (report / complaint / compare / pet vs.)
    template_name = entry.get("template") or "report.html"

    # -----------------------------
    # DİL BİLGİLERİ (ESKİ/YENİ UYUMLU)
    # -----------------------------
    target_lang = ctx.get("target_lang") or ctx.get("targetLanguage") or "tr"
    second_lang = ctx.get("second_lang") or ctx.get("secondLanguage") or ""

    target_lang = str(target_lang).lower()
    second_lang = str(second_lang).lower()

    bilingual_flag = bool(second_lang)
    language_pair = (
        target_lang.upper()
        if not second_lang
        else f"{target_lang.upper()} / {second_lang.upper()}"
    )

    # ctx içinde de normalize edip saklayalım (eski kayıtlar için de ileride lazım olur)
    ctx["target_lang"] = target_lang
    ctx["second_lang"] = second_lang
    ctx["bilingual_flag"] = bilingual_flag
    ctx["language_pair"] = language_pair

    # -----------------------------
    # ANALİZ VERİSİ
    # -----------------------------
    # Eski kayıtlarda analysis ayrı tutulmuş olabilir
    analysis = ctx.get("analysis") or ctx

    # -----------------------------
    # UI METİNLERİ (SOL MENÜ, BUTONLAR) - 1. / 2. DİL
    # -----------------------------
    try:
        ui = build_ui_texts(target_lang=target_lang, second_lang=second_lang)
    except Exception:
        ui = {}

    # -----------------------------
    # RAPOR İÇİ LABEL'LER (BAŞLIK & ETİKETLER) - 1. / 2. DİL
    # -----------------------------
    labels = build_labels(target_lang=target_lang, second_lang=second_lang)

    # -----------------------------
    # MARKA / DETAY / TARİH BİLGİSİ
    # -----------------------------
    brand_label = (
        ctx.get("brand_label")
        or ctx.get("brand")
        or "WellnessAnalyzer"
    )
    detail_label = ctx.get("detail_label") or ctx.get("detail") or "Ultra"
    generated_at = ctx.get("generated_at") or datetime.now().strftime("%d.%m.%Y %H:%M")

    # Template'lere ortak gidecek context
    base_context = {
        "request": request,
        "analysis": analysis,
        "ctx": ctx,
        "ui": ui,
        "labels": labels,
        "brand_label": brand_label,
        "detail_label": detail_label,
        "language_pair": language_pair,
        "generated_at": generated_at,
        "target_lang": target_lang,
        "second_lang": second_lang,
        "report_id": report_id,
    }

    # -----------------------------
    # TEMPLATE SEÇİMİ
    # -----------------------------
    if template_name == "report.html":
        return templates.TemplateResponse("report.html", base_context)

    elif template_name == "complaint_report.html":
        # Şikayet bazlı analiz raporu da aynı dil/label mantığını kullanır
        return templates.TemplateResponse("complaint_report.html", base_context)

    elif template_name == "compare_report.html":
        # İki test karşılaştırma raporu da aynı dil/label mantığını kullanır
        return templates.TemplateResponse("compare_report.html", base_context)

    elif template_name == "petreport.html":
        # PatiPro gibi evcil hayvan raporu varsa, yine aynı context ile çalışsın
        return templates.TemplateResponse("petreport.html", base_context)

    # Bilinmeyen template kaydedilmişse
    raise HTTPException(status_code=500, detail="Bilinmeyen rapor şablonu.")


@app.get("/complaint-report/{report_id}", response_class=HTMLResponse)
async def get_complaint_report(report_id: str, request: Request):
    """
    Şikâyet bazlı analiz raporunu gösteren endpoint.
    /analyze-complaint -> history'ye kaydeder
    /complaint-report/{id} -> bu fonksiyonla ekrandan izlenir.
    """
    entry = get_history_entry(report_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Rapor bulunamadı.")

    ctx = entry.get("ctx", {}) or {}
    if not isinstance(ctx, dict):
        raise HTTPException(status_code=500, detail="Geçersiz rapor kaydı.")

    # Şikayet analizinde analysis zaten ctx["analysis"] içinde
    analysis = ctx.get("analysis") or ctx

    # Diller
    target_lang = ctx.get("target_lang") or ctx.get("targetLanguage") or "tr"
    second_lang = ctx.get("second_lang") or ctx.get("secondLanguage") or ""

    # UI metinleri (sol üstteki buton vb. için)
    try:
        ui = build_ui_texts(target_lang=target_lang, second_lang=second_lang)
    except Exception:
        ui = {}

    # 🔹 labels'ı hazırlayalım (ui içinden ya da boş dict)
    labels = ui.get("labels", {}) if isinstance(ui, dict) else {}

    # Marka etiketi
    brand = ctx.get("brand", "onemore")
    brand_map = {
        "onemore": "OneMore International",
        "xaura": "Xaura Global",
        "atomy": "Atomy",
        "doctorem": "Doctorem",
    }
    brand_label = brand_map.get(str(brand).lower(), str(brand))

    # Dil çifti & tarih
    language_pair = ctx.get("lang_pair") or (
        target_lang.upper()
        if not second_lang
        else f"{target_lang.upper()} / {second_lang.upper()}"
    )
    generated_at = ctx.get("generated_at", "")

    return templates.TemplateResponse(
        "complaint_report.html",
        {
            "request": request,
            "analysis": analysis,
            "ui": ui,
            "labels": labels,              # 🔴 HATA BURADAN ÇÖZÜLDÜ
            "brand_label": brand_label,
            "target_lang": target_lang,
            "second_lang": second_lang,
            "language_pair": language_pair,
            "generated_at": generated_at,
        },
    )

# ====================================================
#  PDF ANALİZ ENDPOINT
# ====================================================

@app.post("/analyze-pdf", response_class=HTMLResponse)
async def analyze_pdf(
    request: Request,
    pdf_file: UploadFile = File(...),
    detail: DetailLevel = Form("ultra"),
    target_lang: str = Form("tr"),
    second_lang: str = Form(""),
    bilingual: str = Form("0"),
    brand: str = Form("onemore"),
):
    # ctx her durumda tanımlı olsun (hata alsak bile)
    ctx: dict = {}

    # 1) İki dil ayarını netleştir (11 dil buradan yönetiliyor)

    # Formdan gelen değeri bool'a çevir
    bilingual_flag = str(bilingual).lower() in ("1", "true", "on", "yes")

    # Kullanıcı iki dilli istemediyse 2. dili tamamen kapat
    if not bilingual_flag:
        second_lang = ""

    # Aynı dili seçtiyse 2. dili iptal et
    if second_lang == target_lang:
        second_lang = ""

    # 2) PDF'i diske kaydet (tmp klasörüne)
    os.makedirs("tmp", exist_ok=True)
    file_path = os.path.join("tmp", pdf_file.filename)

    with open(file_path, "wb") as f:
        f.write(await pdf_file.read())

    try:
        # 3) Cihaz PDF metnini oku ve vücut formu yakala
        pdf_text = read_pdf_text(file_path)
        vucut_form_info = extract_vucut_formu_from_device_pdf(pdf_text)

        # 4) 1. BÖLÜM: GENEL RAPOR
        prompt_part1 = build_analysis_prompt_part1(
            pdf_text,
            detail,
            target_lang,
            second_lang,
            brand,
        )
        analysis = await call_openai_with_prompt(prompt_part1)

        if not isinstance(analysis, dict):
            raise HTTPException(
                status_code=500,
                detail="Analiz çıktısı geçersiz (1. bölüm)",
            )

        # 5) 2. BÖLÜM: 46 SİSTEM KARTI
        prompt_cards = build_system_cards_prompt(
            pdf_text,
            target_lang,
            second_lang,
            brand,
        )
        cards_json = await call_openai_with_prompt(prompt_cards)

        if isinstance(cards_json, dict):
            analysis["sistem_kartlari"] = cards_json.get("sistem_kartlari", [])
        else:
            analysis["sistem_kartlari"] = []

        # 5.5) ÇOCUK İSE YETİŞKİN KARTLARINI SİL
        analysis = filter_child_system_cards_if_needed(analysis)

        # 6) Marka bazlı ürün filtreleme
        brand_key = brand if brand in PRODUCT_LISTS else "onemore"
        analysis = apply_brand_product_filter(analysis, brand_key)

        # 6.5) CİNSİYET KART FİLTRESİ (yetişkin için)
        analysis = apply_gender_card_filter(analysis)

        # 7) Boş kartları doldur
        analysis = fill_empty_system_cards(analysis)

        # 7.5) Cihazdan gelen vücut formu bilgilerini birleştir
        if vucut_form_info:
            analysis.setdefault("vucut_formu", {})
            analysis["vucut_formu"].update(vucut_form_info)

        analysis.setdefault("vucut_formu", {})
        analysis["vucut_formu"].setdefault("etiket", "")
        analysis["vucut_formu"].setdefault("oran", "")
        analysis["vucut_formu"].setdefault("aciklama", "")
        analysis["vucut_formu"].setdefault("bmr_kcal", "")
        analysis["vucut_formu"].setdefault("form_puani", "")

        # 8) ŞU AN İÇİN: make_bilingual_json'u KULLANMIYORUZ
        # Çünkü OpenAI zaten iki satırlı (birincil + ikinci dil) metni üretiyor.

        # 9) UI label'ları ve context hazırla
        detail_label = DETAIL_LEVELS.get(detail, "Ultra Detaylı Analiz (Tek Mod)")

        try:
            ui_texts = build_ui_texts(
                target_lang=target_lang,
                second_lang=second_lang,
            )
        except NameError:
            # build_ui_texts tanımlı değilse
            ui_texts = {}
        except Exception as e:
            # Her ihtimale karşı logla ama sistemi düşürme
            print("build_ui_texts hatası:", e)
            ui_texts = {}

        ctx = {
            "analysis": analysis,
            "detail_label": detail_label,
            "target_lang": target_lang,
            "second_lang": second_lang,
            "bilingual": bool(second_lang),
            "brand": brand_key,
            "brand_label": BRAND_LABELS.get(brand_key, "OneMore International"),
            "ui_texts": ui_texts,
        }

        # 10) Geçmiş kaydı oluştur
        report_id = f"pdf_{int(datetime.utcnow().timestamp() * 1000)}"
        kisi = analysis.get("kisi_bilgileri", {})
        title = kisi.get("ad_soyad") or pdf_file.filename or "PDF Raporu"

        save_history_entry(
            {
                "id": report_id,
                "title": title,
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                "type": "pdf",
                "template": "report.html",
                "ctx": ctx,
            }
        )

        return RedirectResponse(
            url=f"/report/{report_id}",
            status_code=303,
        )

    finally:
        # 🔚 İş bittiğinde tmp içindeki bu PDF'i sil
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            # Sessiz geç, loglamak istersen burada log yazabilirsin
            pass



@app.get("/api/admin/reports", response_class=JSONResponse)
async def api_admin_reports():
    """
    Admin tarafında tüm history kaydını ham haliyle döner.
    Buradan Excel’e atıp, istatistik çıkarılabilir.
    """
    history = load_history()
    return {"items": history}


@app.get("/api/history", response_class=JSONResponse)
async def api_history():
    """
    Kullanıcı tarafı: son 90 gün içindeki tüm raporların
    özet listesini döner (ID, tip, tarih, marka, isim).
    """
    history = load_history()
    now = datetime.utcnow()
    cutoff = now - timedelta(days=HISTORY_MAX_DAYS)

    items: List[Dict[str, Any]] = []

    # En yeni kayıtlar üstte gözüksün diye ters çeviriyoruz
    for item in reversed(history):
        ts = item.get("created_at")
        dt = None
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", ""))
            except Exception:
                dt = None

        # Tarih parse edilemiyorsa, güvenlik için gösterelim
        if dt is not None and dt < cutoff:
            continue

        items.append({
            "id": item.get("id"),
            "type": item.get("type"),
            "brand": item.get("brand"),
            "lang": item.get("lang"),
            "created_at": ts,
            "name": item.get("name"),
            "summary": item.get("summary", ""),
        })

    return {"items": items}


# ====================================================
#  ŞİKAYET BAZLI ANALİZ  (YENİ SÜRÜM)
# ====================================================

@app.post("/analyze-complaint", response_class=HTMLResponse)
async def analyze_complaint(
    request: Request,
    complaint_text: str = Form(...),
    detail: DetailLevel = Form("ultra"),
    target_lang: str = Form("tr"),
    second_lang: str = Form(""),
    bilingual: str = Form("0"),
    brand: str = Form("onemore"),
):
    """
    Serbest metin şikayet analizi.
    - 1 dilli veya 2 dilli çalışır
    - JSON MODE ile OpenAI'den analiz alır
    - Sonucu complaint_report.html ile gösterilmek üzere history'ye kaydeder
    """

    # -----------------------------
    # 1) Dil ayarlarını netleştir
    # -----------------------------
    bilingual_flag = str(bilingual).lower() in ("1", "true", "on", "yes")

    if not bilingual_flag:
        second_lang = ""

    if second_lang and second_lang.lower() == (target_lang or "").lower():
        # Aynı dili iki kere seçmişse, ikinci dili iptal et
        second_lang = ""
        bilingual_flag = False

    primary = (target_lang or "tr").lower().strip()
    secondary = (second_lang or "").lower().strip()

    # -----------------------------
    # 2) Marka ve ürün listesi
    # -----------------------------
    brand_key = brand if brand in PRODUCT_LISTS else "onemore"
    brand_label = BRAND_LABELS.get(brand_key, "OneMore International")
    products = get_brand_products(brand_key)
    products_block = "\n".join(f"- {p}" for p in products)

    # Diğer endpoint’lerle aynı dil kuralını kullanalım
    lang_instruction = build_language_instruction(primary, secondary, source="complaint")

    # -----------------------------
    # 3) PROMPT (tamamen JSON odaklı)
    # -----------------------------
    prompt = f"""
You are a senior wellness analysis expert.

LANGUAGE INSTRUCTION:
{lang_instruction}

Always follow this language rule for EVERY field you generate
(summary, root causes, lifestyle strategy, systems, product reasons, disclaimer).

IMPORTANT:
- The output languages MUST strictly follow the instruction above.
- Do NOT use Turkish in the output unless one of the requested languages
  is actually Turkish (code 'tr').

Work with the following free-text complaint and brand:

Brand: {brand_label}

COMPLAINT TEXT:
----------------
{complaint_text}
----------------

Return ONLY valid JSON in this exact schema:

{{
  "sikayet_ozeti": "…",
  "kok_nedenler": "…",
  "onerilen_yasam_stratejisi": "…",
  "onerilen_onemli_sistemler": [
    {{
      "sistem_adi": "…",
      "aciklama": "…"
    }}
  ],
  "onemore_urun_onerileri": [
    {{
      "urun": "product name, taken ONLY from the brand list below",
      "neden": "…",
      "sure": "…"
    }}
  ],
  "tibbi_sorumluluk": "…"
}}

Brand product list (you MUST choose product names only from here, do not invent new products):

{products_block}

STRICT RULES:
- Output MUST be pure JSON (no Markdown, no ``` fences, no extra text).
- All text fields must obey the LANGUAGE INSTRUCTION above.
"""

    # -----------------------------
    # 4) OpenAI çağrısı
    # -----------------------------
    try:
        raw_analysis = await call_openai_with_prompt(prompt)
    except Exception as e:
        print("analyze_complaint / OpenAI hata:", e)
        raise HTTPException(
            status_code=500,
            detail="Şikayet analizi sırasında hata oluştu."
        )

    # -----------------------------
    # 5) JSON'a güvenli çeviri
    # -----------------------------
    if isinstance(raw_analysis, dict):
        analysis = raw_analysis
    else:
        try:
            analysis = json.loads(raw_analysis)
        except Exception as e:
            print("analyze_complaint / JSON parse hata:", e)
            raise HTTPException(
                status_code=500,
                detail="Şikayet analizi geçersiz formatta döndü."
            )

    if not isinstance(analysis, dict):
        raise HTTPException(
            status_code=500,
            detail="Şikayet analizi geçersiz formatta döndü (dict değil)."
        )

    # Marka ürün filtresi ek güvenlik için (fake isimleri ayıklar)
    try:
        analysis = apply_brand_product_filter(analysis, brand_key)
    except Exception as e:
        print("analyze_complaint / apply_brand_product_filter hata:", e)

    # -----------------------------
    # 6) Context / history kaydı
    # -----------------------------
    lang_pair = primary.upper() if not secondary else f"{primary.upper()} / {secondary.upper()}"

    ctx = {
        "mode": "complaint",
        "brand": brand_key,
        "brand_label": brand_label,
        "detail": str(detail),
        "complaint_text": complaint_text,
        "analysis": analysis,
        "target_lang": primary,
        "second_lang": secondary,
        "bilingual_flag": bool(secondary),
        "lang_pair": lang_pair,
        "generated_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
    }

    report_id = f"cmp_{int(datetime.utcnow().timestamp() * 1000)}"
    title = "Şikayet Analizi"

    save_history_entry({
        "id": report_id,
        "title": title,
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "type": "complaint",
        # ⬇️ Senin HTML dosyanın adı neyse onu kullan:
        "template": "complaint_report.html",
        "ctx": ctx,
    })

    return RedirectResponse(
        url=f"/report/{report_id}",
        status_code=303,
    )



    

# ====================================================
#  KARŞILAŞTIRMA RAPORU
# ====================================================

@app.post("/compare-tests", response_class=HTMLResponse)
async def compare_tests(
    request: Request,
    old_pdf: UploadFile = File(...),
    new_pdf: UploadFile = File(...),
    target_lang: str = Form("tr"),
    second_lang: str = Form(""),
    bilingual: str = Form("0"),
    brand: str = Form("onemore"),
):
    # --- ctx'i en başta tanımla (parametre DEĞİL, lokal değişken) ---
    ctx = None

    # Aynı dili seçmişse ikinci dili iptal et
    if second_lang == target_lang:
        second_lang = ""

    # İki dilli istek gelmediyse ikinci dili tamamen kapat
    bilingual_flag = str(bilingual).lower() in ("1", "true", "on", "yes")
    if not bilingual_flag:
        second_lang = ""

    # --- PDF'leri diske kaydet ---
    os.makedirs("tmp", exist_ok=True)
    old_path = os.path.join("tmp", f"old_{old_pdf.filename}")
    new_path = os.path.join("tmp", f"new_{new_pdf.filename}")

    # UploadFile içeriğini okuyup dosyaya yaz
    old_bytes = await old_pdf.read()
    new_bytes = await new_pdf.read()

    with open(old_path, "wb") as f:
        f.write(old_bytes)
    with open(new_path, "wb") as f:
        f.write(new_bytes)

    try:
        # --- PDF metinlerini oku ---
        old_text = read_pdf_text(old_path)
        new_text = read_pdf_text(new_path)
    finally:
        # --- İŞ BİTİNCE GEÇİCİ PDF DOSYALARINI SİL ---
        for p in (old_path, new_path):
            try:
                os.remove(p)
            except Exception:
                # dosya yoksa / silinemiyorsa sistemi düşürme
                pass

    # --- OpenAI promptunu oluştur ve çağır ---
    prompt = build_compare_prompt(old_text, new_text, target_lang, second_lang, brand)
    analysis = await call_openai_with_prompt(prompt)

    # 🔥 ÖNCE ÇOCUK KART FİLTRESİ
    analysis = filter_child_system_cards_if_needed(analysis)

    # --- Marka ürün filtreleri ---
    brand_key = brand if brand in PRODUCT_LISTS else "onemore"
    analysis = apply_brand_product_filter(analysis, brand_key)

    # 🔥 CİNSİYET KART FİLTRESİ (yetişkinler için)
    analysis = apply_gender_card_filter(analysis)

    # --- Boş kartları doldur ---
    analysis = fill_empty_system_cards(analysis)

    # --- İki dilli istek varsa JSON'u iki dilli hale getir (kullanıyorsan) ---
    if second_lang:
        analysis = await make_bilingual_json(analysis, target_lang, second_lang)

    # --- CONTEXT'i (ctx) HER DURUMDA oluştur ---
    ctx = {
        "analysis": analysis,
        "brand": brand_key,
        "brand_label": BRAND_LABELS.get(brand_key, "OneMore International"),
        "target_lang": target_lang,
        "second_lang": second_lang,  # boş da olabilir, sorun değil
    }

    # Ek güvenlik: ctx bir sebeple oluşmazsa kontrollü hata
    if ctx is None:
        raise HTTPException(status_code=500, detail="Karşılaştırma verisi oluşturulamadı.")

    # --- Raporu geçmişe kaydet ---
    report_id = f"compare_{int(datetime.utcnow().timestamp() * 1000)}"
    title = f"Karşılaştırma Raporu – {BRAND_LABELS.get(brand_key, 'Marka')}"

    history_entry = {
        "id": report_id,
        "title": title,
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "type": "compare",
        "template": "compare_report.html",
        "ctx": ctx,
    }
    save_history_entry(history_entry)

    # --- Kullanıcıyı rapor görüntüleme sayfasına yönlendir ---
    return RedirectResponse(
        url=f"/report/{report_id}",
        status_code=303,
    )
