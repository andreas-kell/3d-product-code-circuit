"""Generate the Stim syndrome-extraction circuit for the 3D product code.

The circuit contains a noiseless preparation round, one noisy round of interleaved X and Z
stabilizer measurements, and a final noisy Z measurement of all data qubits.

Measurement schedule:
    The stabilizers are partitioned into numbered groups ``s`` such that no two stabilizers of
    the same group act on the same data qubit. The data qubits are partitioned into numbered
    groups ``q`` such that no two data qubits of the same group belong to the same stabilizer.
    A stabilizer of group ``s`` interacts with its data qubit of group ``q`` at time step
    ``t = s + q``. Every data qubit therefore sees the stabilizer groups in the same order
    ``0, 1, 2, ...``, which keeps the X and Z measurements compatible. Alternating X and Z
    groups yields an interleaved schedule.

Noise model (all with probability ``error_rate``):
    * ``DEPOLARIZE1`` after every ancilla reset.
    * ``DEPOLARIZE1`` on all data qubits as input noise before the noisy round.
    * ``DEPOLARIZE2`` after every CNOT.
    * ``DEPOLARIZE1`` on data qubits idling during a CNOT layer.
    * Classical flip of every measurement result.
"""

import logging
from collections.abc import Hashable
from functools import reduce
from pathlib import Path

import numpy as np
import stim

logger = logging.getLogger(__name__)

# Elements for creating stabilizers
I = np.array(  # noqa: E741
    [
        [1, 0],
        [0, 1],
    ]
)
h = np.array([1, 1])

# Settings
ERROR_RATE = 1e-3

# Code parameters
DATA_QUBIT_COUNT = 512
DATA_QUBIT_GROUP_COUNT = 8


def get_x_stabilizer_by_group() -> dict[int, np.ndarray]:
    """Get the X stabilizers, keyed by their stabilizer group.

    Each group contains the X stabilizers of one direction of the cube.

    Returns:
        Mapping from stabilizer group to a binary matrix of shape ``(64, 512)``.
    """
    return {
        1: reduce(np.kron, [h, h, h, I, I, I, I, I, I]),
        3: reduce(np.kron, [I, I, I, h, h, h, I, I, I]),
        5: reduce(np.kron, [I, I, I, I, I, I, h, h, h]),
    }


def get_z_stabilizer_by_group() -> dict[int, np.ndarray]:
    """Get the Z stabilizers, keyed by their stabilizer group.

    Returns:
        Mapping from stabilizer group to a binary matrix of shape ``(64, 512)``.
    """
    return {
        0: reduce(np.kron, [h, I, I, h, I, I, h, I, I]),
        2: reduce(np.kron, [I, h, I, I, h, I, I, h, I]),
        4: reduce(np.kron, [I, I, h, I, I, h, I, I, h]),
    }


def get_pcm_x() -> np.ndarray:
    """Get the X parity-check matrix.

    Returns:
        Binary matrix of shape ``(192, 512)``.
    """
    return np.vstack(list(get_x_stabilizer_by_group().values()))


def get_pcm_z() -> np.ndarray:
    """Get the Z parity-check matrix.

    Returns:
        Binary matrix of shape ``(192, 512)``.
    """
    return np.vstack(list(get_z_stabilizer_by_group().values()))


def get_stabilizers_by_group() -> dict[int, tuple[str, np.ndarray]]:
    """Get all stabilizers, keyed by their stabilizer group, in schedule order.

    Returns:
        Mapping from stabilizer group to a tuple of the Pauli type (``"X"`` or ``"Z"``) and
        the binary stabilizer matrix of that group, sorted by stabilizer group.
    """
    stabilizers = {s: ("X", m) for s, m in get_x_stabilizer_by_group().items()}
    stabilizers |= {s: ("Z", m) for s, m in get_z_stabilizer_by_group().items()}
    return dict(sorted(stabilizers.items()))


def get_stim_logical_operators() -> tuple[np.ndarray, np.ndarray]:
    """Get logical operators from stim.

    Returns:
        The X and Z logical operators.
    """
    # Get parity check matrices
    pcm_x = get_pcm_x()
    pcm_z = get_pcm_z()

    # Convert stabilizers to Pauli strings
    stabilizers_x_pauli = [
        stim.PauliString("".join({0: "I", 1: "X"}[v] for v in stab)) for stab in pcm_x
    ]
    stabilizers_z_pauli = [
        stim.PauliString("".join({0: "I", 1: "Z"}[v] for v in stab)) for stab in pcm_z
    ]
    stabilizers_pauli = stabilizers_x_pauli + stabilizers_z_pauli

    # Calculate completed tableau with stim
    completed_tableau = stim.Tableau.from_stabilizers(
        stabilizers_pauli,
        allow_redundant=True,
        allow_underconstrained=True,
    )

    # Retrieve logical operators
    observables = []
    for k in range(len(completed_tableau))[::-1]:
        z = completed_tableau.z_output(k)
        if z in stabilizers_pauli:
            break
        x = completed_tableau.x_output(k)
        observables.append((x, z))

    # Convert to numpy arrays
    logical_x = np.empty((174, 512), dtype=np.uint8)
    logical_z = np.empty((174, 512), dtype=np.uint8)
    for i, observable in enumerate(observables):
        x, z_dummy = observable[0].to_numpy()
        x_dummy, z = observable[1].to_numpy()
        assert x_dummy.sum() == 0
        assert z_dummy.sum() == 0
        logical_x[i] = x
        logical_z[i] = z
    return logical_x, logical_z


def parity3(n: int) -> int:
    """Parity (XOR) of the three bits of n, for n in {0, ..., 7}.

    Args:
        n: Integer in ``{0, ..., 7}``.

    Returns:
        ``n_0 XOR n_1 XOR n_2``.
    """
    return (n ^ (n >> 1) ^ (n >> 2)) & 1


def get_qubit_group(x: int, y: int, z: int) -> int:
    """Direct expression for q(x, y, z).

    The bits of n are denoted by n_0, n_1, n_2, with n_0 the
    least-significant bit.

    Args:
        x: Coordinate in ``{0, ..., 7}``.
        y: Coordinate in ``{0, ..., 7}``.
        z: Coordinate in ``{0, ..., 7}``.

    Returns:
        The data-qubit group in ``{0, ..., 7}``.
    """
    x0 = x & 1
    x1 = (x >> 1) & 1

    y0 = y & 1
    y2 = (y >> 2) & 1

    z1 = (z >> 1) & 1
    z2 = (z >> 2) & 1

    low_bits = (x0 + 2 * x1 + y2 + 2 * y0 + z1 + 2 * z2) % 4
    high_bit = parity3(x) ^ parity3(y) ^ parity3(z)

    return low_bits + 4 * high_bit


def get_qubit_coordinates(qubit_index: int) -> tuple[int, int, int]:
    """Convert a data-qubit index to its cube coordinates.

    Args:
        qubit_index: Data-qubit index in ``{0, ..., 511}``.

    Returns:
        The coordinates ``(x, y, z)``, each in ``{0, ..., 7}``.
    """
    return qubit_index % 8, (qubit_index // 8) % 8, (qubit_index // 8**2) % 8


def get_group_by_qubit() -> np.ndarray:
    """Get the data-qubit group of every data qubit.

    Returns:
        Integer array of shape ``(512,)``.
    """
    return np.array([get_qubit_group(*get_qubit_coordinates(n)) for n in range(DATA_QUBIT_COUNT)])


def get_qubits_by_group() -> np.ndarray:
    """Get qubits by group.

    There are eight groups with 64 qubits each.

    Returns:
        Integer array of shape ``(8, 64)``, where row ``q`` lists the data qubits of group ``q``.
    """
    group_by_qubit = get_group_by_qubit()
    return np.array([np.flatnonzero(group_by_qubit == q) for q in range(DATA_QUBIT_GROUP_COUNT)])


def get_cnot_schedule() -> dict[int, np.ndarray]:
    """Order the data qubits of every stabilizer by their data-qubit group.

    Returns:
        Mapping from stabilizer group ``s`` to an integer array of shape ``(64, 8)``, where
        entry ``[i, q]`` is the data qubit of group ``q`` in stabilizer ``i``. It is measured
        at time step ``t = s + q``.

    Raises:
        ValueError: If a stabilizer does not contain exactly one data qubit per group.
    """
    group_by_qubit = get_group_by_qubit()
    schedule = {}
    for s, (_, stabilizers) in get_stabilizers_by_group().items():
        schedule_s = np.empty((len(stabilizers), DATA_QUBIT_GROUP_COUNT), dtype=int)
        for i, stabilizer in enumerate(stabilizers):
            support = np.flatnonzero(stabilizer)
            groups = group_by_qubit[support]
            if sorted(groups) != list(range(DATA_QUBIT_GROUP_COUNT)):
                raise ValueError(f"Stabilizer {i} of group {s} has data-qubit groups {groups}")
            schedule_s[i, groups] = support
        schedule[s] = schedule_s
    return schedule


def get_ancilla_offsets() -> dict[int, int]:
    """Get the index of the first ancilla qubit of every stabilizer group.

    The ancillas follow the data qubits, ordered by stabilizer group.

    Returns:
        Mapping from stabilizer group to its first ancilla index.
    """
    offsets = {}
    offset = DATA_QUBIT_COUNT
    for s, (_, stabilizers) in get_stabilizers_by_group().items():
        offsets[s] = offset
        offset += len(stabilizers)
    return offsets


class MeasurementRecord:
    """Track measurement indices to create relative ``rec`` targets for detectors.

    Attributes:
        count: Number of measurements registered so far.
    """

    def __init__(self) -> None:
        """Create an empty measurement record."""
        self.count = 0
        self._history: dict[Hashable, list[int]] = {}

    def add(self, key: Hashable) -> None:
        """Register the next measurement in the circuit.

        Args:
            key: Identifier of the measured object, e.g. ``(s, i)`` for a stabilizer.
        """
        self._history.setdefault(key, []).append(self.count)
        self.count += 1

    def target(self, key: Hashable, lookback: int = 0) -> stim.GateTarget:
        """Get a ``rec`` target relative to the current end of the measurement record.

        Args:
            key: Identifier of the measured object.
            lookback: ``0`` for the latest measurement of ``key``, ``1`` for the one before, etc.

        Returns:
            Target to be used in ``DETECTOR`` or ``OBSERVABLE_INCLUDE`` instructions.
        """
        return stim.target_rec(self._history[key][-1 - lookback] - self.count)


def append_syndrome_round(
    circuit: stim.Circuit,
    record: MeasurementRecord,
    error_rate: float,
    add_detectors: bool,
) -> None:
    """Append one round of interleaved X and Z stabilizer measurements.

    Args:
        circuit: Circuit to append to.
        record: Measurement record of ``circuit``, updated in place.
        error_rate: Probability of all noise channels.
        add_detectors: Whether to compare each stabilizer result with the previous round.
    """
    stabilizers_by_group = get_stabilizers_by_group()
    schedule = get_cnot_schedule()
    ancilla_offsets = get_ancilla_offsets()
    qubits_by_group = get_qubits_by_group()

    def ancillas(s: int) -> range:
        return range(ancilla_offsets[s], ancilla_offsets[s] + len(schedule[s]))

    # Reset ancillas
    for s, (pauli, _) in stabilizers_by_group.items():
        circuit.append("R" + pauli, ancillas(s))
        circuit.append("DEPOLARIZE1", ancillas(s), error_rate)
    circuit.append("TICK")

    # Apply CNOTs: stabilizer group s meets data-qubit group q at time step t = s + q
    time_step_count = max(stabilizers_by_group) + DATA_QUBIT_GROUP_COUNT
    for t in range(time_step_count):
        cnot_targets = []
        noise_targets = []
        busy_groups = []
        for s, (pauli, _) in stabilizers_by_group.items():
            q = t - s
            if not 0 <= q < DATA_QUBIT_GROUP_COUNT:
                continue
            busy_groups.append(q)
            for ancilla, data_qubit in zip(ancillas(s), schedule[s][:, q], strict=True):
                control, target = (ancilla, data_qubit) if pauli == "X" else (data_qubit, ancilla)
                cnot_targets += [int(control), int(target)]
                noise_targets += [int(data_qubit), int(ancilla)]
        assert len(busy_groups) == len(set(busy_groups)), "Data qubit used twice in time step"
        circuit.append("CX", cnot_targets)
        circuit.append("DEPOLARIZE2", noise_targets, error_rate)

        idle_qubits = np.sort(
            qubits_by_group[[q for q in range(DATA_QUBIT_GROUP_COUNT) if q not in busy_groups]],
            axis=None,
        )
        if idle_qubits.size:
            circuit.append("DEPOLARIZE1", idle_qubits.tolist(), error_rate)
        circuit.append("TICK")

    # Measure ancillas
    for s, (pauli, _) in stabilizers_by_group.items():
        circuit.append("M" + pauli, ancillas(s), error_rate)
        for i in range(len(ancillas(s))):
            record.add((s, i))
    if add_detectors:
        for s in stabilizers_by_group:
            for i in range(len(ancillas(s))):
                circuit.append(
                    "DETECTOR", [record.target((s, i)), record.target((s, i), lookback=1)]
                )
    circuit.append("TICK")


def append_final_z_measurement(
    circuit: stim.Circuit,
    record: MeasurementRecord,
    error_rate: float,
) -> None:
    """Append the noisy Z measurement of all data qubits, with detectors and observables.

    Args:
        circuit: Circuit to append to.
        record: Measurement record of ``circuit``, updated in place.
        error_rate: Flip probability of the data-qubit measurements.
    """
    circuit.append("MZ", range(DATA_QUBIT_COUNT), error_rate)
    for n in range(DATA_QUBIT_COUNT):
        record.add(("data", n))

    # Compare Z stabilizers with the product of the measured data qubits
    for s, stabilizers in get_z_stabilizer_by_group().items():
        for i, stabilizer in enumerate(stabilizers):
            targets = [record.target((s, i))]
            targets += [record.target(("data", int(n))) for n in np.flatnonzero(stabilizer)]
            circuit.append("DETECTOR", targets)

    # Logical Z observables
    _logical_x_operators, logical_z_operators = get_stim_logical_operators()
    for k, logical_operator in enumerate(logical_z_operators):
        targets = [record.target(("data", int(n))) for n in np.flatnonzero(logical_operator)]
        circuit.append("OBSERVABLE_INCLUDE", targets, k)


def create_circuit(error_rate: float = ERROR_RATE) -> stim.Circuit:
    """Create the syndrome-extraction circuit.

    The data qubits start in ``|0>``, so all Z stabilizers and logical Z operators are
    deterministic. A noiseless round of stabilizer measurements projects onto the code space
    and provides the reference values for the X-stabilizer detectors of the noisy round.

    Args:
        error_rate: Probability of all noise channels.

    Returns:
        The stim circuit.
    """
    logger.info("Creating circuit with error rate %s", error_rate)
    circuit = stim.Circuit()
    record = MeasurementRecord()

    # Initialize data qubits
    circuit.append("R", range(DATA_QUBIT_COUNT))
    circuit.append("TICK")

    # Noiseless initial round
    init_round = stim.Circuit()
    append_syndrome_round(init_round, record, error_rate, add_detectors=False)
    circuit += init_round.without_noise()

    # Input noise
    circuit.append("DEPOLARIZE1", range(DATA_QUBIT_COUNT), error_rate)

    # Noisy round
    append_syndrome_round(circuit, record, error_rate, add_detectors=True)

    # Final measurement
    append_final_z_measurement(circuit, record, error_rate)

    assert record.count == circuit.num_measurements
    return circuit


def main() -> None:
    """Create the circuit, print a summary and check that it has a valid error model."""
    logging.basicConfig(level=logging.INFO)
    circuit = create_circuit()
    print(
        f"qubits={circuit.num_qubits}, measurements={circuit.num_measurements}, "
        f"detectors={circuit.num_detectors}, observables={circuit.num_observables}, "
        f"ticks={circuit.num_ticks}"
    )
    # Raises if any detector or observable is not deterministic in the noiseless circuit
    dem = circuit.detector_error_model()
    print(f"error mechanisms={dem.num_errors}")

    # Save some files to the local directory
    folder = Path("results")
    folder.mkdir(exist_ok=True, parents=True)
    with (folder / "circuit.txt").open("w", encoding="utf-8") as f:
        f.write(str(circuit))
    with (folder / "circuit.svg").open("w", encoding="utf-8") as f:
        f.write(str(circuit.diagram("timeline-svg")))
    with (folder / "detector_error_model.txt").open("w", encoding="utf-8") as f:
        f.write(str(dem))


if __name__ == "__main__":
    main()
