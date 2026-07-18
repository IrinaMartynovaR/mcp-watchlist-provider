import feedparser

from domain.models import Source
from rss_feeds import client
from rss_feeds.classify import classify_item_kind
from rss_feeds.settings import RSSSettings


def test_classify_detects_noise_in_titles() -> None:
    assert classify_item_kind("The best Black Friday deals on SSDs") == "noise"
    assert classify_item_kind("LEGO sets are on sale right now") == "noise"
    assert classify_item_kind("Скидки недели в Steam") == "noise"
    assert classify_item_kind("Промокоды и распродажа в Epic") == "noise"


def test_classify_noise_requires_word_boundary_for_english_markers() -> None:
    assert classify_item_kind("The ideal cozy game for autumn") == "news"
    assert classify_item_kind("An ordeal in the wasteland") == "news"


def test_classify_detects_reviews_by_title_url_and_tags() -> None:
    assert classify_item_kind("Elden Ring review: a triumph") == "review"
    assert classify_item_kind("Обзор Baldur's Gate 3") == "review"
    assert classify_item_kind("Silksong", url="https://example.com/reviews/silksong-verdict") == "review"
    assert classify_item_kind("Dune Part Two", tags=["Reviews", "Movies"]) == "review"


def test_classify_noise_wins_over_review_markers() -> None:
    assert classify_item_kind("Elden Ring review copies on sale") == "noise"


def test_classify_detects_releases() -> None:
    assert classify_item_kind("Hades 2 is out now on PC") == "release"
    assert classify_item_kind("Вышла Hades 2") == "release"
    assert classify_item_kind("Hades 2", summary="The release date is finally here") == "release"


def test_classify_defaults_to_news() -> None:
    assert classify_item_kind("Studio lays off 200 developers") == "news"


def test_clean_summary_strips_html_and_truncates() -> None:
    raw = '<p>Great <b>game</b>&nbsp;&mdash; a hit!</p><img src="https://x.example/pixel.gif"/>'
    assert client._clean_summary(raw, 500) == "Great game — a hit!"
    assert len(client._clean_summary("word " * 200, 500)) <= 500


def test_parse_items_sets_kind_and_cleaned_summary() -> None:
    rss_xml = (
        '<?xml version="1.0"?><rss version="2.0"><channel><title>Feed</title>'
        "<item><title>Elden Ring review</title><link>https://example.com/elden</link>"
        "<description>&lt;p&gt;A &lt;b&gt;masterpiece&lt;/b&gt;&lt;/p&gt;</description></item>"
        "<item><title>Best GPU deals today</title><link>https://example.com/deals</link>"
        "<description>Save big</description></item>"
        "</channel></rss>"
    )
    source = Source.model_validate(
        {"name": "Feed", "category": "games", "language": "en", "url": "https://example.com/rss"}
    )

    items = client._parse_items(feedparser.parse(rss_xml), source=source, limit=10, settings=RSSSettings())

    assert [item.kind for item in items] == ["review", "noise"]
    assert items[0].summary == "A masterpiece"
