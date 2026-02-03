export function createBoard(container, height, width) {
  container.style.gridTemplateColumns = `repeat(${width}, 1fr)`;
  container.innerHTML = "";
  const cells = [];
  for (let r = 0; r < height; r += 1) {
    for (let c = 0; c < width; c += 1) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.dataset.row = String(r);
      cell.dataset.col = String(c);
      container.appendChild(cell);
      cells.push(cell);
    }
  }
  return { container, cells, height, width };
}

export function resetBoard(board) {
  board.cells.forEach((cell) => {
    cell.className = "cell";
  });
}

export function applyStep(board, step) {
  const idx = step.r * board.width + step.c;
  const cell = board.cells[idx];
  if (!cell) {
    return;
  }
  cell.classList.remove("hit", "miss", "sunk");
  if (step.type === "HIT") {
    cell.classList.add("hit");
  } else if (step.type === "MISS") {
    cell.classList.add("miss");
  } else if (step.type === "SUNK") {
    cell.classList.add("sunk");
  }
}
