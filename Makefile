
PROJECT = github.com/drjerry/mfrs

# to override LDFLAGS or CPPFLAGS, supply a Makefile.in
LDFLAGS = -lcblas

ifneq ("$(wildcard Makefile.in)","")
	include Makefile.in
endif

all:
	CGO_LDFLAGS=$(LDFLAGS) go install $(PROJECT)/mfsgd
	CGO_LDFLAGS=$(LDFLAGS) go install $(PROJECT)/mfeval

test:
	go test $(PROJECT)
	CGO_LDFLAGS=$(LDFLAGS) go test $(PROJECT)/linalg
