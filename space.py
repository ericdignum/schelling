"""
Mesa Space Module
=================
Objects used to add a spatial component to a model.
Grid: base grid, a simple list-of-lists.
SingleGrid: grid which strictly enforces one object per cell.
MultiGrid: extension to Grid where each cell is a set of objects.
"""
# Instruction for PyLint to suppress variable name errors, since we have a
# good reason to use one-character variable names for x and y.
# pylint: disable=invalid-name

# Mypy; for the `|` operator purpose
# Remove this __future__ import once the oldest supported Python is 3.10
from __future__ import annotations

import itertools
import math
from warnings import warn

import numpy as np

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Set,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

# For Mypy
from mesa.agent import Agent
from numbers import Real
import numpy.typing as npt

Coordinate = Tuple[int, int]
# used in ContinuousSpace
FloatCoordinate = Union[Tuple[float, float], npt.NDArray[float]]
NetworkCoordinate = int

Position = Union[Coordinate, FloatCoordinate, NetworkCoordinate]

GridContent = Union[Agent, None]
MultiGridContent = List[Agent]

F = TypeVar("F", bound=Callable[..., Any])


def clamp(x: float, lowest: float, highest: float) -> float:
    # This should be faster than np.clip for a scalar x.
    # TODO: measure how much faster this function is.
    return max(lowest, min(x, highest))


def accept_tuple_argument(wrapped_function: F) -> F:
    """Decorator to allow grid methods that take a list of (x, y) coord tuples
    to also handle a single position, by automatically wrapping tuple in
    single-item list rather than forcing user to do it."""

    def wrapper(*args: Any) -> Any:
        if isinstance(args[1], tuple) and len(args[1]) == 2:
            return wrapped_function(args[0], [args[1]])
        else:
            return wrapped_function(*args)

    return cast(F, wrapper)


def is_integer(x: Real) -> bool:
    # Check if x is either a CPython integer or Numpy integer.
    return isinstance(x, (int, np.integer))


class Grid:
    """Base class for a square grid.
    Grid cells are indexed by [x][y], where [0][0] is assumed to be the
    bottom-left and [width-1][height-1] is the top-right. If a grid is
    toroidal, the top and bottom, and left and right, edges wrap to each other
    Properties:
        width, height: The grid's width and height.
        torus: Boolean which determines whether to treat the grid as a torus.
        grid: Internal list-of-lists which holds the grid cells themselves.
    """

    def __init__(self, width: int, height: int, torus: bool) -> None:
        """Create a new grid.
        Args:
            width, height: The width and height of the grid
            torus: Boolean whether the grid wraps or not.
        """
        self.height = height
        self.width = width
        self.torus = torus

        self.grid: List[List[GridContent]] = []

        for x in range(self.width):
            col: List[GridContent] = []
            for y in range(self.height):
                col.append(self.default_val())
            self.grid.append(col)

        # Add all cells to the empties list.
        self.empties = set(itertools.product(*(range(self.width), range(self.height))))

        # Neighborhood Cache
        self._neighborhood_cache: Dict[Any, List[Coordinate]] = dict()

    @staticmethod
    def default_val() -> None:
        """Default value for new cell elements."""
        return None

    @overload
    def __getitem__(self, index: int) -> List[GridContent]:
        ...

    @overload
    def __getitem__(
        self, index: Tuple[int | slice, int | slice]
    ) -> GridContent | List[GridContent]:
        ...

    @overload
    def __getitem__(self, index: Sequence[Coordinate]) -> List[GridContent]:
        ...

    def __getitem__(
        self,
        index: int | Sequence[Coordinate] | Tuple[int | slice, int | slice],
    ) -> GridContent | List[GridContent]:
        """Access contents from the grid."""

        if isinstance(index, int):
            # grid[x]
            return self.grid[index]
        elif isinstance(index[0], tuple):
            # grid[(x1, y1), (x2, y2)]
            index = cast(Sequence[Coordinate], index)

            cells = []
            for pos in index:
                x1, y1 = self.torus_adj(pos)
                cells.append(self.grid[x1][y1])
            return cells

        x, y = index

        if is_integer(x) and is_integer(y):
            # grid[x, y]
            index = cast(Coordinate, index)
            x, y = self.torus_adj(index)
            return self.grid[x][y]

        if is_integer(x):
            # grid[x, :]
            x, _ = self.torus_adj((x, 0))
            x = slice(x, x + 1)

        if is_integer(y):
            # grid[:, y]
            _, y = self.torus_adj((0, y))
            y = slice(y, y + 1)

        # grid[:, :]
        x, y = (cast(slice, x), cast(slice, y))
        cells = []
        for rows in self.grid[x]:
            for cell in rows[y]:
                cells.append(cell)
        return cells

        raise IndexError

    def __iter__(self) -> Iterator[GridContent]:
        """Create an iterator that chains the rows of the grid together
        as if it is one list:"""
        return itertools.chain(*self.grid)

    def coord_iter(self) -> Iterator[Tuple[GridContent, int, int]]:
        """An iterator that returns coordinates as well as cell contents."""
        for row in range(self.width):
            for col in range(self.height):
                yield self.grid[row][col], row, col  # agent, x, y

    def neighbor_iter(self, pos: Coordinate, moore: bool = True) -> Iterator[Agent]:
        """Iterate over position neighbors.
        Args:
            pos: (x,y) coords tuple for the position to get the neighbors of.
            moore: Boolean for whether to use Moore neighborhood (including
                   diagonals) or Von Neumann (only up/down/left/right).
        """

        warn(
            "`neighbor_iter` is deprecated in favor of `iter_neighbors` "
            "and will be removed in the subsequent version."
        )
        return self.iter_neighbors(pos, moore)

    def iter_neighborhood(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> Iterator[Coordinate]:
        """Return an iterator over cell coordinates that are in the
        neighborhood of a certain point.
        Args:
            pos: Coordinate tuple for the neighborhood to get.
            moore: If True, return Moore neighborhood
                        (including diagonals)
                   If False, return Von Neumann neighborhood
                        (exclude diagonals)
            include_center: If True, return the (x, y) cell as well.
                            Otherwise, return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.
        Returns:
            A list of coordinate tuples representing the neighborhood. For
            example with radius 1, it will return list with number of elements
            equals at most 9 (8) if Moore, 5 (4) if Von Neumann (if not
            including the center).
        """
        yield from self.get_neighborhood(pos, moore, include_center, radius)

    def get_neighborhood(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> List[Coordinate]:
        """Return a list of cells that are in the neighborhood of a
        certain point.
        Args:
            pos: Coordinate tuple for the neighborhood to get.
            moore: If True, return Moore neighborhood
                   (including diagonals)
                   If False, return Von Neumann neighborhood
                   (exclude diagonals)
            include_center: If True, return the (x, y) cell as well.
                            Otherwise, return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.
        Returns:
            A list of coordinate tuples representing the neighborhood;
            With radius 1, at most 9 if Moore, 5 if Von Neumann (8 and 4
            if not including the center).
        """
        cache_key = (pos, moore, include_center, radius)
        neighborhood = self._neighborhood_cache.get(cache_key, None)

        if neighborhood is None:
            coordinates: Set[Coordinate] = set()

            x, y = pos
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0 and not include_center:
                        continue
                    # Skip coordinates that are outside manhattan distance
                    if not moore and abs(dx) + abs(dy) > radius:
                        continue

                    coord = (x + dx, y + dy)

                    if self.out_of_bounds(coord):
                        # Skip if not a torus and new coords out of bounds.
                        if not self.torus:
                            continue
                        coord = self.torus_adj(coord)

                    coordinates.add(coord)

            neighborhood = list(coordinates)
            self._neighborhood_cache[cache_key] = neighborhood

        return neighborhood

    def iter_neighbors(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> Iterator[Agent]:
        """Return an iterator over neighbors to a certain point.
        Args:
            pos: Coordinates for the neighborhood to get.
            moore: If True, return Moore neighborhood
                    (including diagonals)
                   If False, return Von Neumann neighborhood
                     (exclude diagonals)
            include_center: If True, return the (x, y) cell as well.
                            Otherwise,
                            return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.
        Returns:
            An iterator of non-None objects in the given neighborhood;
            at most 9 if Moore, 5 if Von-Neumann
            (8 and 4 if not including the center).
        """
        neighborhood = self.get_neighborhood(pos, moore, include_center, radius)
        return self.iter_cell_list_contents(neighborhood)

    def get_neighbors(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> List[Agent]:
        """Return a list of neighbors to a certain point.
        Args:
            pos: Coordinate tuple for the neighborhood to get.
            moore: If True, return Moore neighborhood
                    (including diagonals)
                   If False, return Von Neumann neighborhood
                     (exclude diagonals)
            include_center: If True, return the (x, y) cell as well.
                            Otherwise,
                            return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.
        Returns:
            A list of non-None objects in the given neighborhood;
            at most 9 if Moore, 5 if Von-Neumann
            (8 and 4 if not including the center).
        """
        return list(self.iter_neighbors(pos, moore, include_center, radius))

    def torus_adj(self, pos: Coordinate) -> Coordinate:
        """Convert coordinate, handling torus looping."""
        if not self.out_of_bounds(pos):
            return pos
        elif not self.torus:
            raise Exception("Point out of bounds, and space non-toroidal.")
        else:
            return pos[0] % self.width, pos[1] % self.height

    def out_of_bounds(self, pos: Coordinate) -> bool:
        """Determines whether position is off the grid, returns the out of
        bounds coordinate."""
        x, y = pos
        return x < 0 or x >= self.width or y < 0 or y >= self.height

    @accept_tuple_argument
    def iter_cell_list_contents(
        self, cell_list: Iterable[Coordinate]
    ) -> Iterator[Agent]:
        """Returns an iterator of the contents of the cells
        identified in cell_list.
        Args:
            cell_list: Array-like of (x, y) tuples, or single tuple.
        Returns:
            An iterator of the contents of the cells identified in cell_list
        """
        # Note: filter(None, iterator) filters away an element of iterator that
        # is falsy. Hence, iter_cell_list_contents returns only non-empty
        # contents.
        return filter(None, (self.grid[x][y] for x, y in cell_list))

    @accept_tuple_argument
    def get_cell_list_contents(self, cell_list: Iterable[Coordinate]) -> List[Agent]:
        """Returns a list of the contents of the cells
        identified in cell_list.
        Note: this method returns a list of `Agent`'s; `None` contents are excluded.
        Args:
            cell_list: Array-like of (x, y) tuples, or single tuple.
        Returns:
            A list of the contents of the cells identified in cell_list
        """
        return list(self.iter_cell_list_contents(cell_list))

    def move_agent(self, agent: Agent, pos: Coordinate) -> None:
        """Move an agent from its current position to a new position.
        Args:
            agent: Agent object to move. Assumed to have its current location
                   stored in a 'pos' tuple.
            pos: Tuple of new position to move the agent to.
        """
        pos = self.torus_adj(pos)
        self.remove_agent(agent)
        self._place_agent(agent, pos)
        agent.pos = pos

    def place_agent(self, agent: Agent, pos: Coordinate) -> None:
        """Position an agent on the grid, and set its pos variable."""
        self._place_agent(agent, pos)
        agent.pos = pos

    def _place_agent(self, agent: Agent, pos: Coordinate) -> None:
        """Place the agent at the correct location."""
        x, y = pos
        self.grid[x][y] = agent
        self.empties.discard(pos)

    def remove_agent(self, agent: Agent) -> None:
        """Remove the agent from the grid and set its pos attribute to None."""
        pos = agent.pos
        x, y = pos
        self.grid[x][y] = self.default_val()
        self.empties.add(pos)
        agent.pos = None

    def is_cell_empty(self, pos: Coordinate) -> bool:
        """Returns a bool of the contents of a cell."""
        x, y = pos
        return self.grid[x][y] == self.default_val()

    def move_to_empty(
        self, agent: Agent, cutoff: float = 0.998, num_agents: int | None = None
    ) -> None:
        """Moves agent to a random empty cell, vacating agent's old cell."""
        if len(self.empties) == 0:
            raise Exception("ERROR: No empty cells")
        if num_agents is None:
            try:
                num_agents = agent.model.schedule.get_agent_count()
            except AttributeError:
                raise Exception(
                    "Your agent is not attached to a model, and so Mesa is unable\n"
                    "to figure out the total number of agents you have created.\n"
                    "This number is required in order to calculate the threshold\n"
                    "for using a much faster algorithm to find an empty cell.\n"
                    "In this case, you must specify `num_agents`."
                )
        new_pos = (0, 0)  # Initialize it with a starting value.
        # This method is based on Agents.jl's random_empty() implementation.
        # See https://github.com/JuliaDynamics/Agents.jl/pull/541.
        # For the discussion, see
        # https://github.com/projectmesa/mesa/issues/1052.
        # This switch assumes the worst case (for this algorithm) of one
        # agent per position, which is not true in general but is appropriate
        # here.
        if clamp(num_agents / (self.width * self.height), 0.0, 1.0) < cutoff:
            # The default cutoff value provided is the break-even comparison
            # with the time taken in the else branching point.
            # The number is measured to be 0.998 in Agents.jl, but since Mesa
            # run under different environment, the number is different here.
            while True:
                new_pos = (
                    agent.random.randrange(self.width),
                    agent.random.randrange(self.height),
                )
                if self.is_cell_empty(new_pos):
                    break
        else:
            new_pos = agent.random.choice(list(self.empties))
        self.remove_agent(agent)
        self._place_agent(agent, new_pos)
        agent.pos = new_pos

    def find_empty(self) -> Coordinate | None:
        """Pick a random empty cell."""
        import random

        warn(
            (
                "`find_empty` is being phased out since it uses the global "
                "`random` instead of the model-level random-number generator. "
                "Consider replacing it with having a model or agent object "
                "explicitly pick one of the grid's list of empty cells."
            ),
            DeprecationWarning,
        )

        if self.exists_empty_cells():
            pos = random.choice(list(self.empties))
            return pos
        else:
            return None

    def exists_empty_cells(self) -> bool:
        """Return True if any cells empty else False."""
        return len(self.empties) > 0


class SingleGrid(Grid):
    """Grid where each cell contains exactly at most one object."""

    empties: Set[Coordinate] = set()

    def position_agent(
        self, agent: Agent, x: int | str = "random", y: int | str = "random"
    ) -> None:
        """Position an agent on the grid.
        This is used when first placing agents! Use 'move_to_empty()'
        when you want agents to jump to an empty cell.
        Use 'swap_pos()' to swap agents positions.
        If x or y are positive, they are used, but if "random",
        we get a random position.
        Ensure this random position is not occupied (in Grid).
        """
        if x == "random" or y == "random":
            if len(self.empties) == 0:
                raise Exception("ERROR: Grid full")
            coords = agent.random.choice(list(self.empties))
        else:
            coords = (x, y)
        agent.pos = coords
        self._place_agent(agent, coords)

    def _place_agent(self, agent: Agent, pos: Coordinate) -> None:
        if self.is_cell_empty(pos):
            super()._place_agent(agent, pos)
        else:
            raise Exception("Cell not empty")


class NetworkGrid:
    """Network Grid where each node contains zero or more agents."""

    def __init__(self, G: Any) -> None:
        self.G = G
        for node_id in self.G.nodes:
            G.nodes[node_id]["agent"] = list()

    def place_agent(self, agent: Agent, node_id: int) -> None:
        """Place a agent in a node."""

        self._place_agent(agent, node_id)
        agent.pos = node_id

    def get_neighbors(self, node_id: int, include_center: bool = False) -> List[int]:
        """Get all adjacent nodes"""

        neighbors = list(self.G.neighbors(node_id))
        if include_center:
            neighbors.append(node_id)

        return neighbors

    def move_agent(self, agent: Agent, node_id: int) -> None:
        """Move an agent from its current node to a new node."""

        self.remove_agent(agent)
        self._place_agent(agent, node_id)
        agent.pos = node_id

    def _place_agent(self, agent: Agent, node_id: int) -> None:
        """Place the agent at the correct node."""

        self.G.nodes[node_id]["agent"].append(agent)

    def remove_agent(self, agent: Agent) -> None:
        """Remove the agent from the network and set its pos attribute to None."""
        node_id = agent.pos
        self.G.nodes[node_id]["agent"].remove(agent)
        agent.pos = None

    def is_cell_empty(self, node_id: int) -> bool:
        """Returns a bool of the contents of a cell."""
        return not self.G.nodes[node_id]["agent"]

    def get_cell_list_contents(self, cell_list: List[int]) -> List[GridContent]:
        """Returns the contents of a list of cells ((x,y) tuples)
        Note: this method returns a list of `Agent`'s; `None` contents are excluded.
        """
        return list(self.iter_cell_list_contents(cell_list))

    def get_all_cell_contents(self) -> List[GridContent]:
        """Returns a list of the contents of the cells
        identified in cell_list."""
        return list(self.iter_cell_list_contents(self.G))

    def iter_cell_list_contents(self, cell_list: List[int]) -> List[GridContent]:
        """Returns an iterator of the contents of the cells
        identified in cell_list."""
        list_of_lists = [
            self.G.nodes[node_id]["agent"]
            for node_id in cell_list
            if not self.is_cell_empty(node_id)
        ]
        return [item for sublist in list_of_lists for item in sublist]