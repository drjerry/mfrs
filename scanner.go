package mfrs

import (
	"bufio"
	"fmt"
	"io"
	"regexp"
	"strconv"
)

/* Scanner is a structure for scanning file streams where each line is a
`<row> <col> <value>` triple -- two integers and a float, whitespace delimited.

It adheres to an API like bufio.Scanner in the Go standard libary.
*/
type Scanner struct {
	scanner *bufio.Scanner
	index   [2]int64
	value   float64
	err     error
}

func NewScanner(reader io.Reader) *Scanner {
	return &Scanner{scanner: bufio.NewScanner(reader)}
}

/* Scan reads a record from the underlying file stream. It returns true if
no error is encountered. It returns false if either the end-of-file is reached,
or an error is encountered. In the latter case, Scanner.Err() reports the
error state.
*/
func (s *Scanner) Scan() bool {
	if !s.scanner.Scan() {
		s.err = s.scanner.Err()
		return false
	}
	s.readRecord()
	if s.err != nil {
		return false
	}
	return true
}

// Err returns the error state encountered by Scanner.Scan(), if applicable.
func (s *Scanner) Err() error {
	return s.err
}

// Record returns the data read in the last Scanner.Scan() call.
func (s Scanner) Record() (int, int, float32) {
	return int(s.index[0]), int(s.index[1]), float32(s.value)
}

var wspace = regexp.MustCompile(`\s+`)

func (s *Scanner) readRecord() {
	tokens := wspace.Split(s.scanner.Text(), 4)
	if len(tokens) < 2 {
		s.err = fmt.Errorf("invalid record %s", s.scanner.Text())
		return
	}

	for i := 0; i < 2; i++ {
		s.index[i], s.err = strconv.ParseInt(tokens[i], 0, 32)
		if s.err != nil {
			return
		}
	}
	if len(tokens) >= 3 {
		s.value, s.err = strconv.ParseFloat(tokens[2], 32)
	}
}
