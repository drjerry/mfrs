package main

import (
	"github.com/drjerry/mfrs"
	"github.com/drjerry/mfrs/linalg"
)

// SGDSolver is a type for applying Stochastic Gradient Descent to an
// instance of `mfrs.Model`. It very lightweight in that the primary "state"
// it maintains is references to fields in model.
type SGDSolver struct {
	pbias, qbias         linalg.Vector
	pwts, qwts           linalg.Matrix
	rate, lambda         float32
	dp_i, dq_j, p_i, q_j linalg.Vector
}

// NewSGDSolver constructs a new SGDSolver that will update the `model`.
func NewSGDSolver(model *mfrs.Model, args *Args) *SGDSolver {
	return &SGDSolver{
		pbias:  linalg.Vector(model.Pbias),
		qbias:  linalg.Vector(model.Qbias),
		pwts:   linalg.MatrixView(int(model.Nrow), int(model.Ldim), model.Pwts),
		qwts:   linalg.MatrixView(int(model.Ncol), int(model.Ldim), model.Qwts),
		rate:   args.Rate,
		lambda: args.Lambda,
		dp_i:   linalg.NewVector(int(model.Ldim)),
		dq_j:   linalg.NewVector(int(model.Ldim)),
	}
}

// Update applies one SGD step for the <i, j, r_ij> training example.
// It returns the square-error, (r_ij - h(i,j))^2.
func (s *SGDSolver) Update(i, j int, r_ij float32) float32 {
	s.pwts.RowView(i, &s.p_i)
	s.qwts.RowView(j, &s.q_j)
	e_ij := s.pbias[i] + s.qbias[j] + linalg.Vdot(s.p_i, s.q_j) - r_ij
	// dq_j <- e_{ij} * p_i + lambda * q_j
	s.dq_j.Copy(s.p_i)
	linalg.Vscal(e_ij, s.dq_j)
	linalg.Vaxpy(s.lambda, s.q_j, s.dq_j)
	// dp_i <- e_{ij} * q_j + lambda * p_i
	s.dp_i.Copy(s.q_j)
	linalg.Vscal(e_ij, s.dp_i)
	linalg.Vaxpy(s.lambda, s.p_i, s.dp_i)
	// apply gradient
	s.pbias[i] -= s.rate * e_ij
	s.qbias[j] -= s.rate * e_ij
	linalg.Vaxpy(-s.rate, s.dp_i, s.p_i)
	linalg.Vaxpy(-s.rate, s.dq_j, s.q_j)
	return e_ij * e_ij
}
