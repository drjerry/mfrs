package mfrs

type Rating struct {
    Row, Col int
    Val      float32
}

// Ratings is a simple container for the "ratings" matrix entries r_{ij}.
type Ratings []Rating

// Add appends value, r_{i, j} = val, to the ratings set.
func (r *Ratings) Add(row, col int, val float32) {
    *r = append(*r, Rating{row, col, val})
}
