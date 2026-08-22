"""Hardware/VRAM fit verdict: the data must be sourced, and silence is the
default answer.

Two separate promises are tested here. First, that no VRAM number ever enters
the shipped knowledge floor without a citation - "never synthesize" is a
convention everywhere else in this repo, and a failing test here is the only
thing that makes it binding. Second, that fit_verdict returns None on every
path where it has nothing actionable to say, because a verdict that fires on
the happy path costs tokens on every guidance lookup forever.
"""

from pathlib import Path

import pytest
import yaml

from comfy_draftsman import knowledge

FAMILIES = Path(knowledge.__file__).parent / "families"


def _gb(value: float) -> int:
    return int(value * 1024**3)


def _devices(total_gb: float, free_gb: float | None = None) -> list[dict]:
    free = total_gb if free_gb is None else free_gb
    return [{"name": "cuda:0 Test GPU", "vram_total": _gb(total_gb), "vram_free": _gb(free)}]


# --- the data itself -------------------------------------------------------


@pytest.mark.parametrize("path", sorted(FAMILIES.glob("*.yaml")), ids=lambda p: p.stem)
def test_vram_numbers_carry_a_source(path):
    """A VRAM floor with no citation is a guess, and a guess is worse than the
    `unknown` we would otherwise report."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    hardware = data.get("hardware")
    if not hardware or not hardware.get("vram_gb"):
        return  # no claim, nothing to source
    source = hardware.get("source", "")
    assert source.startswith("http"), f"{path.name}: hardware.vram_gb needs a source URL"
    numbers = hardware["vram_gb"]
    assert set(numbers) <= {"minimum", "recommended"}, numbers
    assert numbers, "an empty vram_gb block claims nothing - drop it instead"
    for key, value in numbers.items():
        assert isinstance(value, (int, float)) and value > 0, f"{path.name}: {key}={value!r}"


def test_families_without_data_stay_silent():
    """The five unsourced families must report nothing rather than a default -
    and an agent asking about them must not be nagged on every call."""
    for family in ("chroma", "krea2", "ltx", "qwen_image", "sd35", "sd15"):
        guidance = knowledge.get_guidance(family)
        assert "hardware" not in guidance, f"{family} gained an unsourced hardware block"
        assert knowledge.fit_verdict(guidance, _devices(4)) is None


# --- the verdict -----------------------------------------------------------


def test_insufficient_when_total_is_below_the_minimum():
    verdict = knowledge.fit_verdict(knowledge.get_guidance("flux"), _devices(8))
    assert verdict["verdict"] == "insufficient"
    assert verdict["required_gb"] == 12
    assert verdict["vram_total_gb"] == 8.0
    assert verdict["source"].startswith("https://")
    # the family's curated prose is folded into the advice, not returned raw
    assert "fp8" in verdict["advice"]


def test_comfortable_fit_is_silent():
    assert knowledge.fit_verdict(knowledge.get_guidance("flux"), _devices(48)) is None
    assert knowledge.fit_verdict(knowledge.get_guidance("sdxl"), _devices(24)) is None


def test_tight_when_free_vram_is_the_only_shortfall():
    """Installed VRAM is fine, another job is holding it. That is a 'free the
    cache' situation, not a model-incompatibility one - and the advice has to
    say which."""
    verdict = knowledge.fit_verdict(knowledge.get_guidance("sdxl"), _devices(24, free_gb=2))
    assert verdict["verdict"] == "tight"
    assert "manage_queue" in verdict["advice"]
    # the model itself fits: the verdict must not claim otherwise
    assert verdict["vram_total_gb"] == 24.0


def test_tight_when_under_the_recommended_but_over_the_minimum():
    verdict = knowledge.fit_verdict(knowledge.get_guidance("flux"), _devices(16))
    assert verdict["verdict"] == "tight"
    assert verdict["required_gb"] == 24


def test_free_vram_shortfall_never_reads_as_insufficient():
    """The regression this ordering exists to prevent: a resident model made
    every fit verdict say 'your GPU cannot run this'."""
    verdict = knowledge.fit_verdict(knowledge.get_guidance("flux"), _devices(48, free_gb=1))
    assert verdict["verdict"] == "tight"


def test_no_devices_is_silent():
    assert knowledge.fit_verdict(knowledge.get_guidance("flux"), []) is None
    # a CPU-only instance reports devices with no usable vram_total
    assert knowledge.fit_verdict(
        knowledge.get_guidance("flux"), [{"name": "cpu", "vram_total": None}]
    ) is None


def test_largest_device_wins_not_the_first():
    """A laptop's iGPU sits at index 0; the render runs on the discrete card."""
    devices = [
        {"name": "cuda:0 iGPU", "vram_total": _gb(2), "vram_free": _gb(2)},
        {"name": "cuda:1 RTX", "vram_total": _gb(48), "vram_free": _gb(48)},
    ]
    assert knowledge.fit_verdict(knowledge.get_guidance("flux"), devices) is None


def test_a_card_reporting_just_under_its_marketing_size_still_fits():
    """Drivers report 7.99GB for an "8GB" card (and some reserve a slice for
    display); without the slack every exactly-spec'd GPU would be told it is
    insufficient for the family it was bought to run."""
    assert knowledge.fit_verdict(knowledge.get_guidance("sdxl"), _devices(7.8)) is None
    # ...but a genuinely smaller card is still called out
    assert knowledge.fit_verdict(knowledge.get_guidance("sdxl"), _devices(6))[
        "verdict"
    ] == "insufficient"


# --- variant overlays ------------------------------------------------------


def test_variant_hardware_overrides_the_family_block(tmp_path):
    """A distilled/quantized variant has its own VRAM floor; get_guidance's
    existing variant merge has to carry `hardware` like everything else."""
    (tmp_path / "flux.yaml").write_text(
        yaml.safe_dump(
            {
                "family": "flux",
                "data": {
                    "variants": {
                        "tiny": {
                            "patterns": ["tinyflux"],
                            # both keys, deliberately: `hardware` deep-merges
                            # like every other variant block (sdxl/turbo restates
                            # all of min/max/default for the same reason), so a
                            # variant that only lowered `minimum` would inherit
                            # the family's `recommended` and still read as tight
                            "hardware": {
                                "vram_gb": {"minimum": 4, "recommended": 6},
                                "source": "https://example.invalid/tiny",
                            },
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    base = knowledge.get_guidance("flux", learned_dir=tmp_path)
    assert base["hardware"]["vram_gb"]["minimum"] == 12

    variant = knowledge.get_guidance("flux", "tinyflux_v1.safetensors", learned_dir=tmp_path)
    assert variant["variant"] == "tiny"
    assert variant["hardware"]["vram_gb"]["minimum"] == 4
    # 8GB is insufficient for base flux but comfortable for the variant
    assert knowledge.fit_verdict(base, _devices(8))["verdict"] == "insufficient"
    assert knowledge.fit_verdict(variant, _devices(8)) is None


def test_bytes_to_gb_tolerates_a_cpu_only_device():
    assert knowledge.bytes_to_gb(None) is None
    assert knowledge.bytes_to_gb(True) is None
    assert knowledge.bytes_to_gb(_gb(16)) == 16.0


# --- the cached-devices path -----------------------------------------------


async def test_cached_devices_never_produce_a_free_vram_verdict(monkeypatch, tmp_path):
    """_State.devices is a process-lifetime snapshot. VRAM *total* cannot change
    while ComfyUI runs, but free VRAM changes constantly - a snapshot taken
    during someone else's render would otherwise keep reporting 'only 1GB free'
    hours after that job finished."""
    from comfy_draftsman import server

    monkeypatch.setattr(server._State, "config", server.Config(session_dir=tmp_path))
    monkeypatch.setattr(
        server._State,
        "devices",
        [{"name": "big", "vram_total": _gb(48), "vram_free": _gb(1)}],
    )
    guidance = knowledge.get_guidance("flux")
    assert server._fit(guidance) is None
    # the same snapshot, read live, is allowed to use it
    verdict = server._fit(guidance, live=True)
    assert verdict["verdict"] == "tight"
    assert "manage_queue" in verdict["advice"]


async def test_cached_devices_still_answer_the_total_vram_question(monkeypatch, tmp_path):
    """Stripping free VRAM must not blind the verdict that actually matters."""
    from comfy_draftsman import server

    monkeypatch.setattr(server._State, "config", server.Config(session_dir=tmp_path))
    monkeypatch.setattr(
        server._State,
        "devices",
        [{"name": "small", "vram_total": _gb(6), "vram_free": _gb(6)}],
    )
    assert server._fit(knowledge.get_guidance("flux"))["verdict"] == "insufficient"
