from dashboard.loaders.local import EpisodeRow, LocalLoader


def _episode(run_index: int) -> EpisodeRow:
    return EpisodeRow(
        task="PiperDynamicPickPlaceTask",
        episode=run_index,
        env_id=0,
        run_index=run_index,
        success=True,
        score=1.0,
        reason=None,
        duration=1.0,
        episode_step=1,
        instruction="Pick up the banana and place it in the box",
        instruction_type="default",
        attributes=[],
        metrics={},
        timing={},
        policy="pi05",
    )


def test_single_env_legacy_videos_match_run_index(tmp_path):
    for idx in range(3):
        (tmp_path / f"Pick_up_the_banana_and_place_it_in_the_box_{idx}.mp4").write_bytes(b"video")
        (tmp_path / f"Pick_up_the_banana_and_place_it_in_the_box_{idx}_viewport.mp4").write_bytes(b"video")

    loader = LocalLoader([])
    for idx in range(3):
        ep = _episode(idx)
        loader._attach_media(tmp_path, ep, has_hdf5_eps=False)
        paths = {v.name: v.path for v in ep.videos}
        assert paths["recording"].endswith(f"_{idx}.mp4")
        assert paths["viewport"].endswith(f"_{idx}_viewport.mp4")

