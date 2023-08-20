#!/usr/bin/env python

import argparse
import cmath
import collections
import fractions
import itertools
import math
import pprint
import sys
import typing

import ase
import ase.build.tools
import ase.io
import numpy as np
import numpy.linalg as la
import numpy.typing as npt


def calc_divisors_of_integer(argument: int) -> typing.Tuple[int, ...]:
    """Find all the divisors of a positive integer.

    This functions proceeds in a way similar to Eratosthenes' sieve, and will
    only work reasonably well for small arguments.

    Args:
        argument: The positive integer whose factors must be found.

    Returns:
        A sorted tuple with all the integer divisors.

    Raises:
        ValueError: If the argument is not a posive integer.
    """
    if argument <= 0:
        raise ValueError("the argument must be positive")
    upper_bound = int(math.sqrt(argument)) + 1
    nruter = [1]
    candidates = list(range(2, upper_bound))
    while candidates:
        candidate = candidates.pop(0)
        if argument % candidate == 0:
            nruter.append(candidate)
            nruter.append(argument // candidate)
        else:
            for multiple in range(2 * candidate, upper_bound, candidate):
                candidates.remove(multiple)
    nruter.append(argument)

    return tuple(sorted(nruter))


def _uniquify_1D(arg: npt.ArrayLike, tolerance: float) -> np.ndarray:
    """Return the unique elements of a 1D array in increasing order.

    Uniqueness is checked down to a tolerance.

    Args:
        arg: The input array.
        tolerance: The maximum distance betweeen two elements that can be
            considered equal.

    Returns:
        A sorted array with the unique elements of the argument, where two
        consecutive elements differ by more than tolerance.
    """
    nruter = np.copy(arg)
    nruter.sort()
    return nruter[np.append(True, np.diff(nruter) > tolerance)]


def check_subperiodicity(atoms: ase.Atoms) -> bool:
    """Check that an Atoms object is periodic only along Z.

    Args:
        atoms: The Atoms object to be checked.

    Returns:
        True if atoms describes a subperiodic system with its periodic axis
        along OZ, and false otherwise.
    """
    cell = atoms.cell[...]
    # Note the zero-tolerance policy on artifactual XZ, YZ, ZY and ZX components
    # of the cell vectors.
    return (
        np.all(cell[2, :2] == 0.0)
        and np.all(cell[:2, 2] == 0.0)
        and tuple(atoms.pbc) == (False, False, True)
    )


class ProjectedElements(typing.NamedTuple):
    """Summary of basic projections used to identify the cyclic group.

    Args:
        axis: The 2D Coordinates of the long axis of the structure.
        lz: The length of the input unit cell of the structure along the long
            axis.
        axis_projections: The projections of all atoms on the long axis, in
            the form of tuples (Z, z), where Z is the atomic number and z the
            corresponding projection. These are sorted first in order of
            increasing Z and then in order of increasing z.
        plane_projections: The projections of all atoms on a plane perpedicular
            to the long axis, with the intersection of the plane and the axis
            taken as the origin of coordinates. Each projection is represented
            as a tuple (Z, w), where Z is the atomic number and w contains the
            two in-plane coordinates stored as a complex number for easier
            operations with both the Cartesian and polar forms. These are sorted
            using the sequence of keys (Z, modulus, theta). Following the usual
            Python convention, theta lies in the range [-pi, pi] and has the
            same sign as the y coordinate.
    """

    axis: npt.ArrayLike
    lz: float
    axis_projections: typing.Tuple[typing.Tuple[int, float], ...]
    plane_projections: typing.Tuple[typing.Tuple[int, complex], ...]


def _calc_center_of_mass(atoms: ase.Atoms) -> np.ndarray:
    """Return the in-plane coordinates of the center of mass of the argument."""
    return atoms.get_center_of_mass()[:2]


def calc_projected_elements(atoms: ase.Atoms) -> ProjectedElements:
    """Build a ProjectedElements object out of a set of atoms representing a
    repeating unit of a quasi-1D nanostructure oriented along the OZ axis.

    Args:
        atoms: The ASE Atoms object to be analyzed.

    Returns:
        A ProjectedElements named tuple; see the docstring of that class for
        details about its contents.

    Raises:
        ValueError: If the periodicity is not correct.
    """
    if not check_subperiodicity(atoms):
        raise ValueError(
            "only quasi-1D systems subperiodic along OZ are accepted"
        )
    axis = _calc_center_of_mass(atoms)
    lz = atoms.cell[2, 2]
    axis_projections = [
        (Z, coords[2])
        for Z, coords in zip(atoms.numbers, atoms.get_positions())
    ]
    axis_projections.sort()
    plane_projections = [
        (Z, complex(coords[0] - axis[0], coords[1] - axis[1]))
        for Z, coords in zip(atoms.numbers, atoms.get_positions())
    ]
    plane_projections.sort(key=lambda x: (x[0], abs(x[1]), cmath.phase(x[1])))
    return ProjectedElements(
        axis, lz, tuple(axis_projections), tuple(plane_projections)
    )


def _test_translation(
    projected_elements: ProjectedElements, delta: float, tolerance: float
) -> bool:
    """Check if a translation is compatible with the axial projections.

    Return True only if a 1-to-1 mapping can be established between the atoms
    and their translated versions as far as their projected positions along OZ
    are concerned. Note that accepted translations need not be compatible with
    the full 3D structure.

    Args:
        projected_elements: The basic projections of atomic positions on the
            long axis and on a perpendicular plane.
        delta: A positive translation, not larger than projected_elements.lz.
        tolerance: The maximum discrepancy between two positions before they
            are considered the same.

    Returns:
        True if the translation is accepted as compatible and False otherwise.
    """
    # The question must be solved element by element.
    by_element = collections.defaultdict(list)
    for el in projected_elements.axis_projections:
        by_element[el[0]].append(el[1])
    for prs in by_element:
        # Perform the translation.
        projections = np.asarray(by_element[prs])
        # Compute the distances between the original and displaces atoms.
        displaced_projections = projections + delta
        dists = (
            projections[:, np.newaxis] - displaced_projections[np.newaxis, :]
        )
        dists -= projected_elements.lz * np.round(
            dists / projected_elements.lz
        )
        dists = np.abs(dists)
        # And try to find a suitable permutation.
        available = list(range(projections.size))
        for src in range(projections.size):
            cand_dists = [dists[src, dst] for dst in available]
            min_pos = np.argmin(cand_dists)
            min_dist = cand_dists[min_pos]
            if min_dist > tolerance:
                return False
            available.remove(available[min_pos])

    return True


def _find_potential_translations(
    projected_elements: ProjectedElements, tolerance: float
) -> np.ndarray:
    """Generate all candidates for the translation part of the cyclic group.

    A translation is represented as a strictly positive number, less than or
    equal than the length of the input unit cell. The translation component of
    the cyclic group is guaranteed to be part of the output, but in general
    there will be many other candidates returned.

    Args:
        projected_elements: The basic projections of atomic positions on the
            long axis and on a perpendicular plane.
        tolerance: The maximum discrepancy between two lengths before they
            are considered the same.

    Returns:
        A NumPy array of unique potential translations.
    """
    # Find the least abundant chemical element.
    by_element = collections.defaultdict(list)
    for el in projected_elements.axis_projections:
        by_element[el[0]].append(el[1])
    projections = np.asarray(
        by_element[min(by_element.keys(), key=lambda x: len(by_element[x]))]
    )
    n_atoms = len(projections)
    # Remove duplicates down to the specified tolerance.
    projections = _uniquify_1D(projections, tolerance)
    # Compute all distinct translations connecting atoms belonging to that
    # element and reduce them to their minimum positive images.
    deltas = projections[:, np.newaxis] - projections[np.newaxis, :]
    deltas -= projected_elements.lz * np.floor(deltas / projected_elements.lz)
    deltas[np.abs(deltas) < tolerance] += projected_elements.lz
    deltas = _uniquify_1D(deltas.ravel(), tolerance)
    # Remove all translations that cannot conceivably generate the right
    # number of atoms.
    possible_order = np.round(projected_elements.lz / deltas).astype(int)
    deltas = deltas[n_atoms % possible_order == 0]
    # Remove all translations that cannot be turned into a valid atomic
    # permutation in this axial analysis.
    return np.asarray(
        [
            d
            for d in deltas
            if _test_translation(projected_elements, d, tolerance)
        ]
    )


def _test_mirror_plane(
    projected_elements: ProjectedElements, theta: complex, tolerance: float
) -> bool:
    """Check if a mirror plane is compatible with the plane projections.

    Return True only if a 1-to-1 mapping can be established between the atoms
    and their mirrored versions as far as their positions projected on the OXY
    place are concerned. Note that accepted operations need not be compatible
    with the full 3D structure.

    Args:
        projected_elements: The basic projections of atomic positions on the
            long axis and on a perpendicular plane.
        theta: An angle with respect to the OX axis describing the mirror plane.
        tolerance: The maximum discrepancy between two positions before they
            are considered the same.

    Returns:
        True if the mirror plane is accepted as compatible and False otherwise.
    """
    # Create an unitary complex number describing the plane and take its square.
    sq_u = cmath.rect(1.0, 2.0 * theta)
    # The question must be solved element by element.
    by_element = collections.defaultdict(list)
    for el in projected_elements.plane_projections:
        by_element[el[0]].append(el[1])
    for prs in by_element:
        # Perform the operation.
        projections = np.asarray(by_element[prs])
        # Compute the distances between the original and displaced atoms.
        mirrored_projections = sq_u * projections.conj()
        dists = np.abs(
            projections[:, np.newaxis] - mirrored_projections[np.newaxis, :]
        )
        # And try to find a suitable permutation.
        available = list(range(projections.size))
        for src in range(projections.size):
            cand_dists = [dists[src, dst] for dst in available]
            min_pos = np.argmin(cand_dists)
            min_dist = cand_dists[min_pos]
            if min_dist > tolerance:
                return False
            available.remove(available[min_pos])

    return True


def _find_potential_vertical_mirror_planes(
    projected_elements: ProjectedElements, tolerance: float
) -> np.ndarray:
    """Generate all candidates for the mirror plane part of the cyclic group.

    A mirror plane is encoded as an in-plane angle in [0, pi), with the zero
    aligned with the OX axis. The reflection component of the glide plane of the
    cyclic group the cyclic group is guaranteed to be part of the output if it
    exists, but in general there will be other candidates returned.

    Args:
        projected_elements: The basic projections of atomic positions on the
            long axis and on a perpendicular plane.
        tolerance: The maximum discrepancy between two lengths before they
            are considered the same.

    Returns:
        A sorted NumPy array of potential mirror planes encoded as angles.
    """
    # If the system is linear, return the empty list: that case should be
    # treated as a rotation.
    r_max = max(abs(e[1]) for e in projected_elements.plane_projections)
    if r_max <= tolerance:
        return []
    # Sort atoms into "shells" composed of atoms from the same element at the
    # same nonzero distance from the origin (within the tolerance), and keep
    # only the least numerous but not empty one.
    smallest_shell = None
    for _, g in itertools.groupby(
        projected_elements.plane_projections, key=lambda x: x[0]
    ):
        lg = list(g)
        rhos = [abs(e[1]) for e in lg]
        beg = 0
        end = 0
        while end < len(lg):
            while end < len(lg) and abs(rhos[end] - rhos[beg]) <= tolerance:
                end += 1
            if (
                rhos[beg] > tolerance
                and smallest_shell is None
                or end - beg < len(smallest_shell)
            ):
                smallest_shell = lg[beg:end]
            beg = end
    # Mirror planes have to pass through an atom or bisect the segment between
    # two.
    nonunique = []
    for i in smallest_shell:
        candidate = cmath.phase(i[1])
        if _test_mirror_plane(projected_elements, candidate, tolerance):
            nonunique.append(candidate % math.pi)
    for i, j in itertools.combinations(smallest_shell, 2):
        candidate = cmath.phase(i[1] + j[1])
        if _test_mirror_plane(
            projected_elements,
            candidate,
            tolerance,
        ):
            nonunique.append(candidate % math.pi)
    nonunique.sort()
    # Prune the list a bit using a conservative criterion based on r_max and
    # tolerance.
    if not nonunique:
        return np.asarray(nonunique)
    nruter = [nonunique[0]]
    for angle in nonunique[1:]:
        dist1 = abs(cmath.rect(r_max, nruter[-1]) - cmath.rect(r_max, angle))
        dist2 = abs(
            cmath.rect(r_max, nruter[-1] + math.pi) - cmath.rect(r_max, angle)
        )
        if min(dist1, dist2) > tolerance:
            nruter.append(angle)
    return np.asarray(nruter)


def _find_potential_horizontal_mirror_planes(
    projected_elements: ProjectedElements, tolerance: float
) -> np.ndarray:
    """Generate all candidate Z coordinates for a horizonzal mirror plane.

    Args:
        projected_elements: The basic projections of atomic positions on the
            long axis and on a perpendicular plane.
        tolerance: The maximum discrepancy between two lengths before they
            are considered the same.

    Returns:
        A NumPy array of potential z coordinates, expressed as positive numbers.
    """
    # Find the least abundant element. A horizontal mirror plane has to
    # pass through one of those atoms or bised the line between two.
    by_element = collections.defaultdict(list)
    for el in projected_elements.axis_projections:
        by_element[el[0]].append(el[1])
    projections = np.asarray(
        by_element[min(by_element.keys(), key=lambda x: len(by_element[x]))]
    )
    projections = _uniquify_1D(projections, tolerance)
    return _uniquify_1D(
        0.5
        * (projections[np.newaxis, :] + projections[:, np.newaxis]).ravel(),
        tolerance,
    )


def test_rototranslation(
    atoms: ase.Atoms, delta: float, angle: float, tolerance: float
) -> typing.Optional[typing.Tuple[int, ...]]:
    """Check if a screw operation is compatible with an Atoms object.

    Args:
        atoms: An ASE atoms object representing a repeating unit of a quasi-1D
            subperiodic system oriented along the OZ axis.
        delta: The length of the translation along the OZ axis.
        angle: The amplitude of the rotation.
        tolerance: The maximum discrepancy between two positions before they
            are considered the same.

    Returns:
        A 1-to-1 mapping between the atoms and their transformed versions, or
        None if no such permutation can be found.
    """
    if not check_subperiodicity(atoms):
        raise ValueError(
            "only quasi-1D systems subperiodic along OZ are accepted"
        )
    atoms = atoms.copy()
    original_positions = atoms.get_positions()
    atoms.translate(np.asarray([0.0, 0.0, delta]))
    atoms.rotate(np.rad2deg(angle), "z", "COM")
    new_positions = atoms.get_positions()
    diffs = new_positions - original_positions[:, np.newaxis, :]

    diffs -= np.einsum(
        "ijk,kl",
        np.round(np.einsum("ijk,kl", diffs, np.linalg.pinv(atoms.cell[...]))),
        atoms.cell[...],
    )
    dists = np.sqrt(np.sum(diffs**2, axis=-1))

    n_atoms = len(atoms)
    atomic_numbers = atoms.numbers
    for src in range(n_atoms):
        dists[src, atomic_numbers != atomic_numbers[src]] = np.infty

    available = list(range(n_atoms))
    permutation = []
    for src in range(n_atoms):
        cand_dists = [dists[src, dst] for dst in available]
        min_pos = np.argmin(cand_dists)
        min_dist = cand_dists[min_pos]
        if min_dist > tolerance:
            return None
        permutation.append(available[min_pos])
        available.remove(available[min_pos])
    return tuple(permutation)


def test_rotoreflection(
    atoms: ase.Atoms, angle: float, tolerance: float
) -> typing.Optional[typing.Tuple[int, ...]]:
    """Check if a rotoreflection operation is compatible with an Atoms object.

    Args:
        atoms: An ASE atoms object representing a repeating unit of a quasi-1D
            subperiodic system oriented along the OZ axis.
        angle: The amplitude of the rotation.
        tolerance: The maximum discrepancy between two positions before they
            are considered the same.

    Returns:
        A 1-to-1 mapping between the atoms and their transformed versions, or
        None if no such permutation can be found.
    """
    if not check_subperiodicity(atoms):
        raise ValueError(
            "only quasi-1D systems subperiodic along OZ are accepted"
        )
    atoms = atoms.copy()
    original_positions = atoms.get_positions()
    atoms.rotate(np.rad2deg(angle), "z", "COM")
    atoms.positions[:, 2] *= -1.0
    new_positions = atoms.get_positions()
    delta = new_positions - original_positions[:, np.newaxis, :]
    delta -= np.einsum(
        "ijk,kl",
        np.round(np.einsum("ijk,kl", delta, np.linalg.pinv(atoms.cell[...]))),
        atoms.cell[...],
    )
    dists = np.sqrt(np.sum(delta**2, axis=-1))

    n_atoms = len(atoms)
    atomic_numbers = atoms.numbers
    for src in range(n_atoms):
        dists[src, atomic_numbers != atomic_numbers[src]] = np.infty

    available = list(range(n_atoms))
    permutation = []
    for src in range(n_atoms):
        cand_dists = [dists[src, dst] for dst in available]
        min_pos = np.argmin(cand_dists)
        min_dist = cand_dists[min_pos]
        if min_dist > tolerance:
            return None
        permutation.append(available[min_pos])
        available.remove(available[min_pos])
    return tuple(permutation)


def test_transflection(
    atoms: ase.Atoms, delta: float, angle: float, tolerance: float
) -> typing.Optional[typing.Tuple[int, ...]]:
    """Check if a glide plane is compatible with an Atoms object.

    Args:
        atoms: An ASE atoms object representing a repeating unit of a quasi-1D
            subperiodic system oriented along the OZ axis.
        delta: The length of the translation along the OZ axis.
        angle: The angle of the glide plane with respect to the OX axis.
        tolerance: The maximum discrepancy between two positions before they
            are considered the same.

    Returns:
        A 1-to-1 mapping between the atoms and their transformed versions, or
        None if no such permutation can be found.
    """
    atoms = atoms.copy()
    original_positions = atoms.get_positions()
    atoms.translate(np.asarray([0.0, 0.0, delta]))
    atoms.rotate(-np.rad2deg(angle), "z", "COM")
    temp_positions = atoms.get_positions()
    y_com = atoms.get_center_of_mass()[1]
    temp_positions[:, 1] = 2.0 * y_com - temp_positions[:, 1]
    atoms.set_positions(temp_positions)
    atoms.rotate(np.rad2deg(angle), "z", "COM")
    new_positions = atoms.get_positions()
    delta = new_positions - original_positions[:, np.newaxis, :]
    delta -= np.einsum(
        "ijk,kl",
        np.round(np.einsum("ijk,kl", delta, np.linalg.pinv(atoms.cell[...]))),
        atoms.cell,
    )
    dists = np.sqrt(np.sum(delta**2, axis=-1))

    n_atoms = len(atoms)
    atomic_numbers = atoms.numbers
    for src in range(n_atoms):
        dists[src, atomic_numbers != atomic_numbers[src]] = np.infty

    available = list(range(n_atoms))
    permutation = []
    for src in range(n_atoms):
        cand_dists = [dists[src, dst] for dst in available]
        min_pos = np.argmin(cand_dists)
        min_dist = cand_dists[min_pos]
        if min_dist > tolerance:
            return None
        permutation.append(available[min_pos])
        available.remove(available[min_pos])
    return tuple(permutation)


def calc_eigenvector_criterion(vector, tensor_of_inertia):
    unit = vector / np.sqrt(np.dot(vector, vector))
    transformed = tensor_of_inertia @ unit
    transformed_unit = transformed / np.sqrt(np.dot(transformed, transformed))
    remainder = unit - transformed_unit
    return np.sqrt(np.dot(remainder, remainder))


def find_OZ_rotations(
    monomer_atoms: ase.Atoms, tolerance: float
) -> np.ndarray:
    """Find all rotations around the OZ axis that leave the monomer invariant.

    Args:
        monomer_atoms: The object containing the atoms in the monomer for a
            particular choice of cyclic group.
        tolerance: The maximum discrepancy between two positions before they
            are considered the same.

    Returns:
        A sorted NumPy array of unique potential rotations encoded as integers.
    """
    center_of_mass = monomer_atoms.get_center_of_mass()
    centered_positions = (
        monomer_atoms.get_positions() - center_of_mass[np.newaxis, :]
    )
    tensor_of_inertia = sum(
        m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        for m, r in zip(monomer_atoms.get_masses(), centered_positions)
    )
    total_mass = monomer_atoms.get_masses().sum()
    radius_of_gyration = np.sqrt(tensor_of_inertia.trace() / total_mass)
    discriminant = radius_of_gyration * calc_eigenvector_criterion(
        np.asarray([0.0, 0.0, 1.0]), tensor_of_inertia
    )
    # If OZ is not an eigenvector of the tensor inertia, no rotations are
    # possible.
    if discriminant >= tolerance:
        return np.asarray([])

    projected_elements = calc_projected_elements(monomer_atoms)
    r_max = max(abs(e[1]) for e in projected_elements.plane_projections)
    if r_max <= tolerance:
        return np.ones(1) * np.infty
    # Sort atoms into "shells" composed of atoms from the same element at the
    # same nonzero distance from the origin (within the tolerance), and keep
    # only the least numerous but not empty one.
    smallest_shell = None
    for _, g in itertools.groupby(
        projected_elements.plane_projections, key=lambda x: x[0]
    ):
        lg = list(g)
        rhos = [abs(e[1]) for e in lg]
        beg = 0
        end = 0
        while end < len(lg):
            while end < len(lg) and abs(rhos[end] - rhos[beg]) <= tolerance:
                end += 1
            if (
                rhos[beg] > tolerance
                and smallest_shell is None
                or end - beg < len(smallest_shell)
            ):
                smallest_shell = lg[beg:end]
            beg = end
    # Similar to translations, possible rotations are built from the
    # differences in angle between atoms in a single shell.
    positions = np.asarray([e[1] for e in smallest_shell])
    rotations = (positions[np.newaxis, :] / positions[:, np.newaxis]).ravel()
    nonunique = np.angle(rotations) % (2.0 * np.pi)
    nonunique.sort()
    # The maximum order of a rotation is equal to the number of atoms in the
    # shell.
    max_order = positions.shape[0]
    rational = set(
        fractions.Fraction(angle / 2.0 / np.pi).limit_denominator(max_order)
        for angle in nonunique
    )
    # Discard the trivial rotation and keep only the smaller rotations as candidates
    return np.asarray(
        sorted([r.denominator for r in rational if r > 0 and r.numerator == 1])
    )


def test_pi_rotation(
    monomer_atoms: ase.Atoms, axis: npt.ArrayLike, tolerance: float
) -> bool:
    """Test if a rotation of pi around an axis is a symmetry of the monomer."""
    center_of_mass = monomer_atoms.get_center_of_mass()
    centered_atoms = monomer_atoms.copy()
    centered_atoms.translate(-center_of_mass)
    transformed_atoms = centered_atoms.copy()
    transformed_atoms.rotate(180.0, axis)
    original_positions = monomer_atoms.get_positions()
    transformed_positions = transformed_atoms.get_positions()

    # TODO: Put this kind into a separate function.
    delta = transformed_positions - original_positions[:, np.newaxis, :]
    dists = np.sqrt(np.sum(delta**2, axis=-1))

    n_atoms = len(monomer_atoms)
    atomic_numbers = monomer_atoms.numbers
    for src in range(n_atoms):
        dists[src, atomic_numbers != atomic_numbers[src]] = np.infty

    available = list(range(n_atoms))
    for src in range(n_atoms):
        cand_dists = [dists[src, dst] for dst in available]
        min_pos = np.argmin(cand_dists)
        min_dist = cand_dists[min_pos]
        if min_dist > tolerance:
            return False
        available.remove(available[min_pos])
    return True


def test_perpendicular_mirror_plane(
    monomer_atoms: ase.Atoms, tolerance: float
) -> bool:
    """Test whether the monomer has a mirror symmetry normal to OZ."""
    projected_elements = calc_projected_elements(monomer_atoms)
    potential_positions = _find_potential_horizontal_mirror_planes(
        projected_elements, tolerance
    )

    l_z = atoms.cell[2, 2]
    n_atoms = len(monomer_atoms)

    for trial_z in potential_positions:
        centered_positions = monomer_atoms.get_positions()
        centered_positions[:, 2] -= trial_z
        transformed_positions = centered_positions.copy()
        transformed_positions[:, 2] *= -1.0

        delta = transformed_positions - centered_positions[:, np.newaxis, :]

        delta[..., 2] -= l_z * np.round(delta[..., 2] / l_z)
        dists = np.sqrt(np.sum(delta**2, axis=-1))

        atomic_numbers = monomer_atoms.numbers
        for src in range(n_atoms):
            dists[src, atomic_numbers != atomic_numbers[src]] = np.infty

        available = list(range(n_atoms))
        for src in range(n_atoms):
            cand_dists = [dists[src, dst] for dst in available]
            min_pos = np.argmin(cand_dists)
            min_dist = cand_dists[min_pos]
            if min_dist > tolerance:
                break
            available.remove(available[min_pos])
        else:
            return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Try to detect the line group of a system"
    )
    parser.add_argument(
        "--tolerance",
        "-t",
        help="minimum absolute distance at which"
        " two positions are considered the same",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "filename", help="path to the file from which coordinates will be read"
    )
    parser.add_argument(
        "-c",
        "--corner",
        help="assume that the input file has the long axis at (x, y)=(0, 0)"
        " and needs to be recentered before partially removing the PBCs",
        action="store_true",
    )
    args = parser.parse_args()

    # TODO: Make it clear that the third axis is assumed to be OZ.
    atoms = ase.io.read(args.filename)
    if args.corner:
        direct_coordinates = atoms.get_scaled_positions()
        direct_coordinates[:, :2] += 0.5
        direct_coordinates[:, :2] %= 1.0
        atoms.set_scaled_positions(direct_coordinates)
    atoms.pbc[:2] = False

    projected_elements = calc_projected_elements(atoms)

    potential_translations = _find_potential_translations(
        projected_elements, args.tolerance
    )
    potential_mirror_planes = _find_potential_vertical_mirror_planes(
        projected_elements, args.tolerance
    )

    element_counter = collections.Counter(atoms.numbers)
    n_least_abundant = min(element_counter.values())
    print("POSSIBLE ORDERS OF THE CYCLIC GROUP:")
    print(calc_divisors_of_integer(n_least_abundant))

    print("POTENTIAL TRANSLATIONS:")
    print(potential_translations)
    print("POTENTIAL MIRROR PLANES:")
    print(potential_mirror_planes)

    # Find A, the shortest pure translation, in case the input structure is
    # a supercell.
    pure_translations = []
    for trans in potential_translations:
        if test_rototranslation(atoms, trans, 0.0, args.tolerance):
            pure_translations.append(trans)
    print("PURE TRANSLATIONS:")
    print(pure_translations)
    translational_period = min(pure_translations)
    print("TRANSLATIONAL PERIOD:", translational_period)

    # Discard any potential translations longer than A.
    potential_translations = potential_translations[
        potential_translations <= translational_period
    ]

    candidate_generators = set()
    # For each surviving candidate translation
    for trans in potential_translations:
        # Find the possible orders of a rotation
        numerator = int(round(translational_period / trans))
        for denominator in range(1, numerator + 1):
            angle = (2.0 * math.pi / (numerator / float(denominator))) % (
                2.0 * math.pi
            )
            # Try and see which ones work.
            if test_rototranslation(atoms, trans, angle, args.tolerance):
                order = fractions.Fraction(
                    numerator=numerator, denominator=denominator
                )
                code = f"C({order})"
                candidate_generators.add((code, trans, angle))
        # Try the glide planes as well.
        for angle in potential_mirror_planes:
            if test_transflection(atoms, trans, angle, args.tolerance):
                code = f"σ({numerator}, {angle})"
                candidate_generators.add((code, trans, angle))

    print("CANDIDATE GENERATORS OF THE CYCLIC GROUP:")
    for g in candidate_generators:
        print(g)

    # Detect the monomer in each case.
    monomers = dict()
    for g in candidate_generators:
        code = g[0]
        transformed_atoms = atoms.copy()
        if code.startswith("C"):
            transposed_orbits = []
            total_permutation = tuple(range(len(atoms)))
            while True:
                new_permutation = test_rototranslation(
                    transformed_atoms, g[1], g[2], args.tolerance
                )
                total_permutation = tuple(
                    total_permutation[i] for i in new_permutation
                )
                transposed_orbits.append(total_permutation)
                if total_permutation == tuple(range(len(atoms))):
                    break
                transformed_atoms = ase.build.tools.sort(
                    transformed_atoms, tags=new_permutation
                )
        else:
            transposed_orbits = [
                test_transflection(atoms, g[1], g[2], args.tolerance),
                tuple(range(len(atoms))),
            ]
        orbits = set(frozenset(t) for t in zip(*transposed_orbits))
        orbits = [list(o) for o in orbits]
        # TODO: Try to find the most compact monomer.
        monomers[code] = sorted([o[0] for o in orbits])

    print("MONOMERS:")
    for c in monomers:
        print(f"{c}: {monomers[c]}")

    # Detect the symmetries of the monomers.
    point_groups = {}
    for c in monomers:
        print(c)
        # Rotations around the OZ axis.
        monomer_atoms = atoms[tuple(i for i in monomers[c])]
        centered_atoms = monomer_atoms.copy()
        centered_atoms.center(about=0.0)
        potential_rotations = np.asarray(
            [
                order
                for order in find_OZ_rotations(centered_atoms, args.tolerance)
                if test_rototranslation(
                    monomer_atoms, 0.0, 2.0 * math.pi / order, args.tolerance
                )
            ]
        )
        potential_rotoreflections = np.asarray(
            [
                order
                for order in find_OZ_rotations(centered_atoms, args.tolerance)
                if test_rotoreflection(
                    monomer_atoms, 2.0 * math.pi / order, args.tolerance
                )
            ]
        )
        print("Rotations around OZ:", potential_rotations)
        print("Rotoreflections around OZ:", potential_rotoreflections)
        # TODO: Deal with linear molecules.
        if potential_rotations.size != 0:
            rotation_order = potential_rotations.max()
        else:
            rotation_order = 1
        if potential_rotoreflections.size != 0:
            rotoreflection_order = potential_rotoreflections.max()
        else:
            rotoreflection_order = 1
        print("Rotation order:", rotation_order)
        print("Rotoreflection order:", rotoreflection_order)

        # Rotations of order 2 around an axis perpendicular to OZ.
        total_mass = monomer_atoms.get_masses().sum()
        center_of_mass = monomer_atoms.get_center_of_mass()
        centered_positions = (
            monomer_atoms.get_positions() - center_of_mass[np.newaxis, :]
        )
        tensor_of_inertia = sum(
            m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
            for m, r in zip(monomer_atoms.get_masses(), centered_positions)
        )
        radius_of_gyration = np.sqrt(tensor_of_inertia.trace() / total_mass)

        eigvecs = la.eigh(tensor_of_inertia)[1]
        pi_rotations = []
        for e in eigvecs.T:
            if abs(e[2]) * radius_of_gyration < args.tolerance:
                if test_pi_rotation(monomer_atoms, e, args.tolerance):
                    pi_rotations.append(e)
                    print("Perpendicular rotation axis:", e)

        # FIXME: Not working for 12x12
        # Mirror plane perpendicular to OZ.
        perpendicular_mirror_plane = test_perpendicular_mirror_plane(
            monomer_atoms, args.tolerance
        )
        if perpendicular_mirror_plane:
            print("Mirror plane perpendicular to OZ")

        # Mirror planes containing OZ.
        centered_atoms = monomer_atoms.copy()
        centered_atoms.translate(-center_of_mass)
        projected_elements = calc_projected_elements(centered_atoms)
        candidate_axial_mirror_planes = _find_potential_vertical_mirror_planes(
            projected_elements, args.tolerance
        )
        axial_mirror_planes = []

        # Test these mirror planes on the monomer,
        # always around the center of mass.
        for candidate in candidate_axial_mirror_planes:
            if test_transflection(
                monomer_atoms, 0.0, candidate, args.tolerance
            ):
                axial_mirror_planes.append(candidate)
                print("Axial mirror plane:", candidate)

        # Identify the point group of the monomer.
        if rotoreflection_order == 2 * rotation_order:
            point_groups[c] = f"S{rotoreflection_order}"
        else:
            if perpendicular_mirror_plane and axial_mirror_planes:
                point_groups[c] = f"D{rotation_order}h"
            elif pi_rotations:
                if axial_mirror_planes:
                    point_groups[c] = f"D{rotation_order}d"
                else:
                    point_groups[c] = f"D{rotation_order}"
            elif axial_mirror_planes:
                point_groups[c] = f"C{rotation_order}v"
            elif perpendicular_mirror_plane:
                point_groups[c] = f"C{rotation_order}h"
            else:
                point_groups[c] = f"C{rotation_order}"

    print("-" * 79)
    print("FACTORS OF THE GROUP")
    print(args.filename)
    pprint.pprint(point_groups)
    print("-" * 79)

    # TODO: Check the effect on the other monomers. Remove operations if
    # necessary.
