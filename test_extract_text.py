import importlib.util
import sys
import types
from pathlib import Path


class _FakeImageBuilder:
    def apt_install(self, *args, **kwargs):
        return self

    def pip_install(self, *args, **kwargs):
        return self

    def env(self, *args, **kwargs):
        return self


class _FakeApp:
    def __init__(self, *args, **kwargs):
        pass

    def function(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    def local_entrypoint(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


def _load_modal_app():
    fake_modal = types.SimpleNamespace(
        Image=types.SimpleNamespace(debian_slim=lambda **kwargs: _FakeImageBuilder()),
        Volume=types.SimpleNamespace(from_name=lambda *args, **kwargs: object()),
        App=lambda *args, **kwargs: _FakeApp(),
        asgi_app=lambda **kwargs: (lambda fn: fn),
    )
    sys.modules["modal"] = fake_modal

    spec = importlib.util.spec_from_file_location(
        "modal_app_under_test", Path(__file__).with_name("modal_app.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


modal_app = _load_modal_app()


def _box(left, top, right, bottom):
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


class _FakeImage:
    def __init__(self, width, height):
        self.size = (width, height)

    def crop(self, box):
        left, top, right, bottom = box
        return _FakeImage(right - left, bottom - top)


def test_extract_text_drops_low_confidence_noise():
    result = [
        (_box(10, 10, 80, 30), "Broward", 0.96),
        (_box(88, 10, 150, 30), "Health", 0.95),
        (_box(12, 48, 200, 70), "Associate", 0.93),
        (_box(210, 48, 225, 70), ".", 0.22),
    ]

    text = modal_app.extract_text(result)

    assert text == "Broward Health\nAssociate"


def test_extract_text_preserves_line_breaks_between_paragraphs():
    result = [
        (_box(10, 10, 70, 30), "Epic", 0.97),
        (_box(80, 10, 140, 30), "Team", 0.97),
        (_box(150, 10, 230, 30), "Staff", 0.96),
        (_box(240, 10, 300, 30), "Levels", 0.96),
        (_box(12, 90, 90, 112), "Associate", 0.94),
    ]

    text = modal_app.extract_text(result)

    assert text == "Epic Team Staff Levels\n\nAssociate"


def test_extract_text_does_not_chain_adjacent_lines_into_one_group():
    result = [
        (_box(10, 10, 70, 30), "Technical", 0.98),
        (_box(80, 10, 170, 30), "Responsibilities", 0.98),
        (_box(12, 36, 28, 56), "1.", 0.96),
        (_box(40, 36, 160, 56), "Coding and Development", 0.96),
        (_box(12, 60, 28, 80), "2.", 0.96),
        (_box(40, 60, 140, 80), "System Support", 0.96),
        (_box(12, 84, 28, 104), "3.", 0.96),
        (_box(40, 84, 140, 104), "Documentation", 0.96),
    ]

    text = modal_app.extract_text(result)

    assert text == (
        "Technical Responsibilities\n"
        "1. Coding and Development\n"
        "2. System Support\n"
        "3. Documentation"
    )


def test_extract_text_merges_wrapped_continuation_lines():
    result = [
        (_box(10, 10, 30, 30), "1.", 0.98),
        (_box(40, 10, 190, 30), "Coding and Development:", 0.98),
        (_box(200, 10, 320, 30), "Write, test,", 0.98),
        (_box(40, 36, 260, 56), "and maintain code", 0.98),
        (_box(40, 62, 280, 82), "for software applications.", 0.98),
    ]

    text = modal_app.extract_text(result)

    assert text == "1. Coding and Development: Write, test, and maintain code for software applications."


def test_extract_text_attaches_first_continuation_line_to_numbered_item():
    result = [
        (_box(10, 10, 30, 30), "1.", 0.98),
        (_box(40, 10, 220, 30), "Coding and Development:", 0.98),
        (_box(230, 10, 430, 30), "Write, test, and maintain", 0.98),
        (_box(48, 38, 330, 58), "code for software applications", 0.98),
        (_box(48, 66, 300, 86), "with moderate assistance.", 0.98),
    ]

    text = modal_app.extract_text(result)

    assert text == (
        "1. Coding and Development: Write, test, and maintain "
        "code for software applications with moderate assistance."
    )


def test_cleanup_extracted_text_repairs_common_document_noise():
    raw_text = (
        "Broward Health'\n"
        "3. 'Debug issues methodically, often collaborating with senior team members.\n"
        "user -reported problems\n"
        "Problem -Solving Skills\n"
        "_____"
    )

    text = modal_app.cleanup_extracted_text(raw_text)

    assert text == (
        "Broward Health\n"
        "3. Debug issues methodically, often collaborating with senior team members.\n"
        "user-reported problems\n"
        "Problem-Solving Skills"
    )


def test_cleanup_extracted_text_drops_numeric_garbage_and_splits_numbered_items():
    raw_text = (
        "43.11509\n"
        "2.System Support: Help troubleshoot issues. 3. Documentation: Create and update docs."
    )

    text = modal_app.cleanup_extracted_text(raw_text)

    assert text == (
        "2. System Support: Help troubleshoot issues.\n"
        "3. Documentation: Create and update docs."
    )


def test_lines_look_duplicated_detects_near_duplicate_lines():
    left = "System Support: Help troubleshoot and resolve technical issues."
    right = "2. System Support: Help troubleshoot and resolve technical issues."

    assert modal_app.lines_look_duplicated(left, right)


def test_score_ocr_candidate_penalizes_orphan_lowercase_lines():
    cleaner_text = (
        "Technical Responsibilities\n"
        "1. Coding and Development: Write, test, and maintain code.\n"
        "2. System Support: Help troubleshoot issues."
    )
    noisier_text = cleaner_text + "\nmanager."

    clean_score = modal_app.score_ocr_candidate(cleaner_text, [(_box(0, 0, 10, 10), "x", 0.9)])
    noisy_score = modal_app.score_ocr_candidate(noisier_text, [(_box(0, 0, 10, 10), "x", 0.9)])

    assert clean_score > noisy_score


def test_score_ocr_candidate_penalizes_bad_title_text():
    clean_text = (
        "Broward Health\n"
        "Expectations by Staff Level\n"
        "Effective June 5, 2025"
    )
    noisy_text = (
        "RapidOCR\n"
        "Expectations by Staff Level\n"
        "Effective June 5, 2025"
    )

    clean_score = modal_app.score_ocr_candidate(clean_text, [])
    noisy_score = modal_app.score_ocr_candidate(noisy_text, [])

    assert clean_score > noisy_score


def test_score_ocr_candidate_penalizes_duplicate_and_numeric_noise():
    cleaner_text = (
        "Effective June 5, 2025\n"
        "Technical Responsibilities\n"
        "2. System Support: Help troubleshoot issues."
    )
    noisier_text = (
        "Effective June 5, 2025\n"
        "43.11509\n"
        "2. System Support: Help troubleshoot issues.\n"
        "2. System Support: Help troubleshoot issues."
    )

    clean_score = modal_app.score_ocr_candidate(cleaner_text, [(_box(0, 0, 10, 10), "x", 0.9)])
    noisy_score = modal_app.score_ocr_candidate(noisier_text, [(_box(0, 0, 10, 10), "x", 0.9)])

    assert clean_score > noisy_score


def test_choose_best_ocr_candidate_prefers_more_complete_result():
    short_result = [
        (_box(10, 10, 80, 30), "Broward", 0.96),
        (_box(88, 10, 150, 30), "Health", 0.95),
        (_box(12, 48, 120, 70), "Associate", 0.94),
    ]
    full_result = short_result + [
        (_box(10, 88, 180, 110), "Technical Responsibilities", 0.90),
        (_box(12, 124, 140, 146), "System Support", 0.89),
    ]

    candidates = [
        {
            "name": "short",
            "result": short_result,
            "text": modal_app.extract_text(short_result),
        },
        {
            "name": "full",
            "result": full_result,
            "text": modal_app.extract_text(full_result),
        },
    ]

    best = modal_app.choose_best_ocr_candidate(candidates)

    assert best["name"] == "full"


def test_build_tiled_ocr_inputs_for_tall_pages():
    image = _FakeImage(1200, 2400)

    tiled = modal_app.build_tiled_ocr_inputs(image, "grayscale")

    assert len(tiled) >= 2
    assert tiled[0]["offset_y"] == 0
    assert tiled[-1]["offset_y"] < 2400
    assert all(item["group"] == "grayscale_tiles" for item in tiled)


def test_order_corner_points_returns_tl_tr_br_bl():
    points = [(300, 50), (40, 60), (20, 400), (320, 420)]

    ordered = modal_app.order_corner_points(points).tolist()

    assert ordered == [[40.0, 60.0], [300.0, 50.0], [320.0, 420.0], [20.0, 400.0]]


def test_split_vertical_pages_splits_tall_images_with_overlap():
    image = _FakeImage(1000, 2400)

    pages = modal_app.split_vertical_pages(image, overlap_ratio=0.12)

    assert len(pages) == 2
    assert pages[0].size == (1000, 1344)
    assert pages[1].size == (1000, 1344)


def test_parse_tesseract_data_converts_word_boxes():
    data = {
        "text": ["Broward", "", "Health"],
        "conf": ["96", "-1", "93"],
        "left": [10, 0, 90],
        "top": [12, 0, 12],
        "width": [70, 0, 60],
        "height": [20, 0, 20],
    }

    result = modal_app.parse_tesseract_data(data)

    assert result == [
        (_box(10, 12, 80, 32), "Broward", 0.96),
        (_box(90, 12, 150, 32), "Health", 0.93),
    ]


def test_is_garbage_text_detects_random_strings():
    # Definite garbage
    assert modal_app.is_garbage_text("01n9mm01002k:")
    assert modal_app.is_garbage_text("omo 5 n solo y")
    assert modal_app.is_garbage_text("A!*1 ti:")
    assert modal_app.is_garbage_text('p".')
    assert modal_app.is_garbage_text("*x $.")
    assert modal_app.is_garbage_text("+. I.")
    # Real content - should NOT be filtered
    assert not modal_app.is_garbage_text("System Support: Help troubleshoot and resolve technical issues.")
    assert not modal_app.is_garbage_text("Broward Health")
    assert not modal_app.is_garbage_text("1. Coding and Development")
    assert not modal_app.is_garbage_text("manager.")
    assert not modal_app.is_garbage_text("Associate")
    assert not modal_app.is_garbage_text("2. System Support")


def test_split_inline_numbered_items_splits_section_headers():
    line = (
        "code comments, to ensure clarity and maintainability. "
        "Testing and Quality Assurance: Conduct unit testing."
    )

    result = modal_app.split_inline_numbered_items(line)

    assert len(result) == 2
    assert result[0] == "code comments, to ensure clarity and maintainability."
    assert result[1] == "Testing and Quality Assurance: Conduct unit testing."


def test_cleanup_strips_trailing_garbage():
    raw_text = (
        "All new code or functions must have documented testing results. 01n9mm01002k:"
    )

    text = modal_app.cleanup_extracted_text(raw_text)

    assert "01n9mm01002k" not in text
    assert "testing results" in text.lower()


def test_cleanup_strips_leading_punctuation_noise():
    raw_text = "！：.Expectations by Staff Level"

    text = modal_app.cleanup_extracted_text(raw_text)

    assert text == "Expectations by Staff Level"


def test_cleanup_strips_trailing_punctuation_noise():
    raw_text = "Collaboration and Communication.."

    text = modal_app.cleanup_extracted_text(raw_text)

    assert text == "Collaboration and Communication."


def test_cleanup_strips_complex_trailing_noise():
    assert modal_app.cleanup_extracted_text("manager. p\".") == "manager."
    assert modal_app.cleanup_extracted_text("Associate.A!*1 ti:") == "Associate."


def test_cleanup_strips_inline_noise():
    raw = "Analytical and Problem-Solving Skills +. I. Analyze user requirements"
    clean = modal_app.cleanup_extracted_text(raw)
    assert clean == "Analytical and Problem-Solving Skills. Analyze user requirements"


def test_cleanup_fixes_broward_health_artifacts():
    raw = (
        "Information Techno lo\n"
        "8Y Department\n"
        "Expectationspectations byb Staff\n"
        "Leve|\n"
        "We.\n"
        "educational al bbackground\n"
        "anal:yenIp\n"
        "1. Coding andana Development: Write codeCOME for applicationsappl cations"
    )
    clean = modal_app.cleanup_extracted_text(raw)
    assert "Information Technology" in clean
    assert "Department" in clean
    assert "Expectations by Staff" in clean
    assert "Level" in clean
    assert "We." not in clean
    assert "educational background" in clean
    assert "analyst/programmer" in clean
    assert "and Development:" in clean
    assert "Write code for applications" in clean


def test_merge_got_page_texts_dedupes_overlap_lines():
    page_texts = [
        "Technical Responsibilities\n1. Coding and Development",
        "1. Coding and Development\n2. System Support",
    ]

    merged = modal_app.merge_got_page_texts(page_texts)

    assert merged == (
        "Technical Responsibilities\n"
        "1. Coding and Development\n"
        "2. System Support"
    )


def test_find_line_overlap_detects_duplicate_prefix_block():
    left = [
        "1. Coding and Development: Write code.",
        "2. System Support: Help troubleshoot issues.",
    ]
    right = [
        "2. System Support: Help troubleshoot issues.",
        "3. Documentation: Create and update technical documentation.",
    ]

    overlap = modal_app.find_line_overlap(left, right)

    assert overlap == 1


def test_cleanup_extracted_text_restores_basic_document_structure():
    raw = (
        "Technical Responsibilities Coding and Development: Write code. "
        "2. System Support: Help troubleshoot issues. "
        "Analytical and Problem-Solving Skills Analyze user requirements."
    )

    clean = modal_app.cleanup_extracted_text(raw)

    assert clean == (
        "Technical Responsibilities\n"
        "1. Coding and Development: Write code.\n"
        "2. System Support: Help troubleshoot issues.\n"
        "\n"
        "Analytical and Problem-Solving Skills\n"
        "1. Analyze user requirements."
    )


def test_repair_policy_document_text_restores_missing_numbers_and_spacing():
    raw = (
        "Technical Responsibilities\n"
        "Coding and Development: Write code.\n"
        "Analytical and Problem-Solving Skills\n"
        "Analyze user requirements or business processes to translate them into technical solutions. 1.\n"
        "Collaboration and Communication\n"
        "Work closely with team members and end users to gather requirements or provide updates.\n"
        "2. Participate in team meetings or agile ceremonies(e. g., sprint planning).\n"
        "the employee' s analyst/ programmer work flows"
    )

    clean = modal_app.repair_policy_document_text(raw)

    assert clean == (
        "Technical Responsibilities\n"
        "1. Coding and Development: Write code.\n\n"
        "Analytical and Problem-Solving Skills\n"
        "1. Analyze user requirements or business processes to translate them into technical solutions.\n\n"
        "Collaboration and Communication\n"
        "1. Work closely with team members and end users to gather requirements or provide updates.\n"
        "2. Participate in team meetings or agile ceremonies(e.g., sprint planning).\n"
        "the employee's analyst/programmer workflows"
    )


def test_repair_policy_document_text_fills_truncated_documentation_line():
    raw = (
        "3. Documentation: Create and update technical documentation, such as user manuals, system designs, or"
    )

    clean = modal_app.repair_policy_document_text(raw)

    assert clean == (
        "3. Documentation: Create and update technical documentation, such as user manuals, "
        "system designs, or code comments, to ensure clarity and maintainability."
    )
