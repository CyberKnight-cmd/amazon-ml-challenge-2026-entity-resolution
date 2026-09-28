from entity_resolution.normalize import fold, norm_addr, norm_name, squash


def test_fold_strips_latin_accents_but_keeps_indic():
    assert fold("Collège Cathare") == "college cathare"
    assert fold("HART VALLEY LÍBERTY") == "hart valley liberty"
    assert fold("अल टेक") == "अल टेक".lower()
    assert fold("प्राइवेट") == "प्राइवेट"  # vowel signs survive


def test_name_punctuation_and_ampersand():
    assert norm_name("Brown & Warfield LLC")[0] == "brown and warfield llc"
    assert norm_name("*** THE BROWN & WARFIELD LLC")[0] == "the brown and warfield llc"
    assert norm_name("Kimble, Olva S., L.C.S.W., (PC)")[0] == "kimble olva s l c s w pc"


def test_name_dedupes_repeated_tokens_and_bracket_noise():
    assert norm_name("Heritage Heritage Semiconductor")[0] == "heritage semiconductor"
    assert norm_name("CARDIOLOGY SAFE CARE INC [INC]")[0] == "cardiology safe care inc"


def test_domain_style_names():
    assert norm_name("kimb1eolvas.com") == ("kimb1eolvas", True)
    assert norm_name("cardiology-safe-care.com") == ("cardiologysafecare", True)
    assert norm_name("Acme Robotics.com")[1] is False  # has spaces -> ordinary name


def test_address_drops_missing_components():
    assert norm_addr("60 Virginia Street, N/A, Terrell, Texas") == "60 virginia street terrell texas"
    assert norm_addr("HOMELAND AVE, NULL, NORMAN, OK") == "homeland ave norman ok"
    assert norm_addr("") == ""
    assert norm_addr("nan") == ""


def test_indic_address_keeps_script_tokens():
    assert "महाराष्ट्र" in norm_addr("Plot No. A-19, Dombivli, महाराष्ट्र").split()


def test_squash():
    assert squash("kimble olva") == "kimbleolva"


def test_address_leading_zeros_stripped():
    assert norm_addr("0019553 Fifteenth Avenue, Shoreline") == "19553 fifteenth avenue shoreline"
    assert norm_addr("Plot 007, Sector 0") == "plot 7 sector 0"
