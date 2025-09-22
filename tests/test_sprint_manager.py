from douglas.sprint_manager import SprintManager


def test_per_sprint_cadence_runs_only_on_final_day():
    manager = SprintManager(sprint_length_days=3)

    decision = manager.should_run_step('demo', {'frequency': 'per_sprint'})
    assert decision.should_run is False

    manager.current_day = 3
    decision = manager.should_run_step('demo', {'frequency': 'per_sprint'})
    assert decision.should_run is True
    assert decision.event_type == 'sprint'

    manager.record_step_execution('demo', decision.event_type)
    again = manager.should_run_step('demo', {'frequency': 'per_sprint'})
    assert again.should_run is False


def test_feature_events_are_consumed_per_step():
    manager = SprintManager(sprint_length_days=5)
    manager.record_commit('feat: first feature')

    decision = manager.should_run_step('feature_refinement', {'frequency': 'per_feature'})
    assert decision.should_run is True
    assert decision.event_type == 'feature'

    manager.record_step_execution('feature_refinement', decision.event_type)
    follow_up = manager.should_run_step('feature_refinement', {'frequency': 'per_feature'})
    assert follow_up.should_run is False

    manager.record_commit('feat(ui): second feature')
    third = manager.should_run_step('feature_refinement', {'frequency': 'per_feature'})
    assert third.should_run is True


def test_push_policy_per_feature_consumes_events():
    manager = SprintManager(sprint_length_days=5)
    manager.record_commit('feat: add capability')

    decision = manager.should_run_push('per_feature')
    assert decision.should_run is True
    assert decision.event_type == 'feature'

    manager.record_push(decision.event_type, 'per_feature')
    after_push = manager.should_run_push('per_feature')
    assert after_push.should_run is False


def test_push_policy_per_sprint_waits_for_last_day():
    manager = SprintManager(sprint_length_days=2)
    manager.record_commit('feat: early work')

    early = manager.should_run_push('per_sprint')
    assert early.should_run is False

    manager.current_day = 2
    ready = manager.should_run_push('per_sprint')
    assert ready.should_run is True
    assert ready.event_type == 'sprint'

    manager.record_push(ready.event_type, 'per_sprint')
    repeat = manager.should_run_push('per_sprint')
    assert repeat.should_run is False


def test_pr_policy_tracks_commits():
    manager = SprintManager(sprint_length_days=4)
    manager.record_commit('fix: resolve issue')

    decision = manager.should_open_pr('per_bug')
    assert decision.should_run is True
    assert decision.event_type == 'bug'

    manager.record_pr(decision.event_type, 'per_bug')
    follow_up = manager.should_open_pr('per_bug')
    assert follow_up.should_run is False
