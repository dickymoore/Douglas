import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.cadence_manager import CadenceManager, should_run_step
from douglas.sprint_manager import SprintManager


def _build_manager(config, sprint_length=4):
    sprint = SprintManager(sprint_length_days=sprint_length)
    cadence = CadenceManager(config, sprint)
    return sprint, cadence


def test_daily_developer_cadence_runs_each_iteration():
    config = {'Developer': {'development': 'daily'}}
    sprint, cadence = _build_manager(config, sprint_length=2)

    decision_day_one = cadence.evaluate_step(
        'generate', {'name': 'generate', 'role': 'Developer', 'activity': 'development'}
    )
    assert decision_day_one.should_run is True

    sprint.record_step_execution('generate', decision_day_one.event_type)
    sprint.finish_iteration()

    decision_day_two = cadence.evaluate_step(
        'generate', {'name': 'generate', 'role': 'Developer', 'activity': 'development'}
    )
    assert decision_day_two.should_run is True


def test_per_sprint_retrospective_waits_for_final_day():
    config = {'ScrumMaster': {'retrospective': 'per_sprint'}}
    sprint, cadence = _build_manager(config, sprint_length=3)

    early = cadence.evaluate_step(
        'retrospective',
        {'name': 'retrospective', 'role': 'ScrumMaster', 'activity': 'retrospective'},
    )
    assert early.should_run is False
    assert 'not scheduled until the end of the sprint' in early.reason.lower()

    sprint.current_day = sprint.sprint_length_days
    final = cadence.evaluate_step(
        'retrospective',
        {'name': 'retrospective', 'role': 'ScrumMaster', 'activity': 'retrospective'},
    )
    assert final.should_run is True
    sprint.record_step_execution('retrospective', final.event_type)

    follow_up = cadence.evaluate_step(
        'retrospective',
        {'name': 'retrospective', 'role': 'ScrumMaster', 'activity': 'retrospective'},
    )
    assert follow_up.should_run is False


def test_per_feature_review_runs_when_feature_available():
    config = {'Developer': {'code_review': 'per_feature'}}
    sprint, cadence = _build_manager(config, sprint_length=5)

    first = cadence.evaluate_step(
        'review', {'name': 'review', 'role': 'Developer', 'activity': 'code_review'}
    )
    assert first.should_run is False
    assert 'waiting for completed features' in first.reason

    sprint.record_commit('feat: initial capability')
    pending = cadence.evaluate_step(
        'review', {'name': 'review', 'role': 'Developer', 'activity': 'code_review'}
    )
    assert pending.should_run is True
    sprint.record_step_execution('review', pending.event_type)

    follow_up = cadence.evaluate_step(
        'review', {'name': 'review', 'role': 'Developer', 'activity': 'code_review'}
    )
    assert follow_up.should_run is False


def test_on_demand_cadence_skips_until_triggered():
    config = {'Stakeholder': {'status_update': 'on_demand'}}
    sprint, cadence = _build_manager(config)

    decision = cadence.evaluate_step(
        'status_update',
        {'name': 'status_update', 'role': 'Stakeholder', 'activity': 'status_update'},
    )
    assert decision.should_run is False
    assert 'manual trigger' in decision.reason.lower()


def test_should_run_step_helper_exposes_context():
    config = {'Developer': {'development': 'daily'}}
    sprint, cadence = _build_manager(config)

    decision = cadence.evaluate_step(
        'generate', {'name': 'generate', 'role': 'Developer', 'activity': 'development'}
    )
    assert decision.should_run is True
    context = cadence.last_context
    assert context is not None
    assert context.sprint_day == 1
    assert context.available_events['feature'] == 0

    assert should_run_step(context.role, context.activity, context) is True
    assert context.decision is not None
    assert 'developer cadence for development' in context.decision.reason.lower()


def test_step_level_cadence_overrides_role_defaults():
    sprint, cadence = _build_manager({}, sprint_length=3)

    early = cadence.evaluate_step('demo', {'name': 'demo', 'cadence': 'per_sprint'})
    assert early.should_run is False

    sprint.current_day = sprint.sprint_length_days
    final_day = cadence.evaluate_step('demo', {'name': 'demo', 'cadence': 'per_sprint'})
    assert final_day.should_run is True
    assert final_day.event_type == 'sprint'

    sprint.record_step_execution('demo', final_day.event_type)
    repeat = cadence.evaluate_step('demo', {'name': 'demo', 'cadence': 'per_sprint'})
    assert repeat.should_run is False


def test_per_epic_activity_waits_for_epic_completion():
    config = {'ProductOwner': {'backlog_refinement': 'per_epic'}}
    sprint, cadence = _build_manager(config, sprint_length=4)

    initial = cadence.evaluate_step('refine', {'name': 'refine'})
    assert initial.should_run is False
    assert 'epic' in initial.reason.lower()

    sprint.mark_epic_completed('EPIC-42')
    ready = cadence.evaluate_step('refine', {'name': 'refine'})
    assert ready.should_run is True
    assert ready.event_type == 'epic'

    sprint.record_step_execution('refine', ready.event_type)
    follow_up = cadence.evaluate_step('refine', {'name': 'refine'})
    assert follow_up.should_run is False
