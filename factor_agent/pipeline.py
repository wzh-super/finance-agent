"""Six explicit research stages, without dynamic dispatch or framework inheritance.

state.current.stage identifies the next action. Each completed stage is saved;
service errors preserve the checkpoint, while failed experiments enter history.
Only factors accepted by research feedback enter the factor library.
"""

import time

from .errors import ExperimentFailed
from .models import Experiment, Feedback
from .progress import metric_comparison


class Pipeline:
    def __init__(self, config, store, researcher, coder, backtester):
        self.config = config
        self.store = store
        self.researcher = researcher
        self.coder = coder
        self.backtester = backtester

    def run(self, rounds: int, *, stop_after_stage: str | None = None):
        """Run until `rounds` total experiments finish; resume skips saved stages."""
        if rounds < 0:
            raise ValueError("rounds must be non-negative")
        stages = {"propose", "design", "code", "backtest", "review", "commit"}
        if stop_after_stage is not None and stop_after_stage not in stages:
            raise ValueError(f"Unknown stop stage: {stop_after_stage}")
        state = self.store.load(self.config.fingerprint())
        self.store.round_id = state.current.round_id if state.current else None
        self.store.event("run", "Run state loaded", completed_rounds=len(state.history), target_rounds=rounds,
                         resume_stage=state.current.stage if state.current else None,
                         accepted_factors=[f.spec.name for f in state.accepted_factors])
        if len(state.history) >= rounds:
            self.store.event("run", "Target round count already reached; returning saved results", report=self.store.root / "report.md")
            return state

        # Compute the fixed baseline once. Backtest caching also covers a crash
        # between worker completion and this state save.
        if state.best_result is None:
            self.store.save(state)
            self.store.event("baseline", "Preparing the fixed LightGBM baseline; checking existing cache first")
            state.best_result = self.backtester.run([], [], self.store.root / "baseline")
            self.store.event("baseline", "Baseline ready", metrics=metric_comparison(state.best_result.metrics),
                             artifacts=state.best_result.workspace)
            self.store.save(state)

        while len(state.history) < rounds:
            if state.current is None:
                state.current = Experiment(round_id=len(state.history))
                self.store.save(state)
            experiment = state.current
            stage = experiment.stage
            directory = self.store.round_dir(experiment.round_id)
            labels = {"propose": "Generate hypothesis", "design": "Define factors", "code": "Generate, execute and repair code",
                      "backtest": "Compute full-data factors, train the fixed model and backtest",
                      "review": "Compare results and evaluate the hypothesis", "commit": "Save the decision and update the factor library"}
            started = time.monotonic()
            self.store.event(stage, f"Round {experiment.round_id} started: {labels[stage]}")

            try:
                if stage == "propose":
                    experiment.hypothesis = self.researcher.propose(state.history)
                    self.store.event(stage, "Hypothesis generated", hypothesis=experiment.hypothesis.hypothesis,
                                     reason=experiment.hypothesis.reason)
                    experiment.stage = "design"
                elif stage == "design":
                    experiment.specs = self.researcher.design(
                        experiment.hypothesis, state.history, state.accepted_factors)
                    if not experiment.specs:
                        raise ExperimentFailed("No new factor tasks remain after design/deduplication")
                    self.store.event(stage, "Factor definitions generated", count=len(experiment.specs))
                    for index, spec in enumerate(experiment.specs, 1):
                        self.store.event(stage, f"Factor {index}/{len(experiment.specs)}: {spec.name}",
                                         description=spec.description, formula=spec.formulation, variables=spec.variables)
                    experiment.stage = "code"
                elif stage == "code":
                    experiment.factors = self.coder.develop(experiment.specs, directory / "coding")
                    if not any(f.implemented for f in experiment.factors):
                        raise ExperimentFailed("No factor passed implementation review")
                    self.store.event(stage, "Implementation reviews complete", implemented=[f.spec.name for f in experiment.factors if f.implemented],
                                     failed={f.spec.name: f.error for f in experiment.factors if not f.implemented})
                    experiment.stage = "backtest"
                elif stage == "backtest":
                    experiment.result = self.backtester.run(
                        state.accepted_factors, [f for f in experiment.factors if f.implemented], directory / "backtest")
                    self.store.event(stage, "Backtest complete: comparison with the best result so far",
                                     metrics=metric_comparison(experiment.result.metrics, state.best_result.metrics),
                                     tested_factors=experiment.result.factor_names,
                                     dropped_factors=experiment.result.dropped_factors,
                                     artifacts=experiment.result.workspace)
                    experiment.stage = "review"
                elif stage == "review":
                    experiment.feedback = self.researcher.review(
                        experiment.hypothesis, experiment.factors, experiment.result, state.best_result, state.history)
                    if type(experiment.feedback.decision) is not bool:
                        raise ExperimentFailed("Research feedback decision must be a boolean")
                    feedback = experiment.feedback
                    self.store.event(stage, "Research feedback generated", decision=feedback.decision,
                                     observations=feedback.observations, hypothesis_evaluation=feedback.hypothesis_evaluation,
                                     reasoning=feedback.reason, next_hypothesis=feedback.new_hypothesis)
                    experiment.stage = "commit"
                elif stage == "commit":
                    self._commit(state)
                    self.store.event(stage, "Round decision finalized; updating the factor library", research_accepted=experiment.feedback.decision,
                                     accepted_factors=[f.spec.name for f in state.accepted_factors],
                                     experiment=directory / "experiment.json", trajectory=self.store.root / "trajectory.md")
            except ExperimentFailed as exc:
                experiment.error = str(exc)
                experiment.feedback = Feedback(str(exc), "Experiment could not be evaluated", "", "Execution failure", False)
                experiment.stage = "commit"
                self.store.event("rejected", str(exc))

            # ServiceError, KeyboardInterrupt and unexpected errors deliberately
            # propagate. Their last completed stage remains safely on disk.
            self.store.save(state)
            self.store.event(stage, f"Round {experiment.round_id} stage finished",
                             elapsed_seconds=round(time.monotonic() - started, 3), next_stage=experiment.stage)
            if stage == stop_after_stage:
                return state
        return state

    @staticmethod
    def _commit(state):
        experiment = state.current
        if experiment.feedback is None:
            raise ValueError("Cannot commit an experiment without feedback")
        if experiment.feedback.decision:
            if experiment.result is None or not experiment.result.added_factors:
                raise ValueError("Accepted experiment must have a backtest and surviving new factors")
            existing = {f.spec.name for f in state.accepted_factors}
            added = experiment.result.added_factors
            if len({f.spec.name for f in added}) != len(added) or existing.intersection(f.spec.name for f in added):
                raise ValueError("Accepted factor library would contain duplicate names")
            state.accepted_factors.extend(added)
            state.best_result = experiment.result
        experiment.stage = "complete"
        state.history.append(experiment)
        state.current = None
