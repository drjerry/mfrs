// Code generated by protoc-gen-go.
// source: proto/model.proto
// DO NOT EDIT!

/*
Package mfrs is a generated protocol buffer package.

It is generated from these files:
	proto/model.proto

It has these top-level messages:
	Model
*/
package mfrs

import proto "github.com/golang/protobuf/proto"
import fmt "fmt"
import math "math"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

type Model struct {
	Ldim  int32     `protobuf:"varint,1,opt,name=ldim" json:"ldim,omitempty"`
	Nrow  int32     `protobuf:"varint,2,opt,name=nrow" json:"nrow,omitempty"`
	Ncol  int32     `protobuf:"varint,3,opt,name=ncol" json:"ncol,omitempty"`
	Pvals []float32 `protobuf:"fixed32,4,rep,name=pvals" json:"pvals,omitempty"`
	Qvals []float32 `protobuf:"fixed32,5,rep,name=qvals" json:"qvals,omitempty"`
}

func (m *Model) Reset()                    { *m = Model{} }
func (m *Model) String() string            { return proto.CompactTextString(m) }
func (*Model) ProtoMessage()               {}
func (*Model) Descriptor() ([]byte, []int) { return fileDescriptor0, []int{0} }

func init() {
	proto.RegisterType((*Model)(nil), "mfrs.Model")
}

var fileDescriptor0 = []byte{
	// 109 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x09, 0x6e, 0x88, 0x02, 0xff, 0xe2, 0x12, 0x2c, 0x28, 0xca, 0x2f,
	0xc9, 0xd7, 0xcf, 0xcd, 0x4f, 0x49, 0xcd, 0xd1, 0x03, 0xb3, 0x85, 0x58, 0x72, 0xd3, 0x8a, 0x8a,
	0x95, 0xfc, 0xb9, 0x58, 0x7d, 0x41, 0x82, 0x42, 0x3c, 0x5c, 0x2c, 0x39, 0x29, 0x99, 0xb9, 0x12,
	0x8c, 0x0a, 0x8c, 0x1a, 0xac, 0x20, 0x5e, 0x5e, 0x51, 0x7e, 0xb9, 0x04, 0x13, 0x9c, 0x97, 0x9c,
	0x9f, 0x23, 0xc1, 0x0c, 0xe6, 0xf1, 0x72, 0xb1, 0x16, 0x94, 0x25, 0xe6, 0x14, 0x4b, 0xb0, 0x28,
	0x30, 0x6b, 0x30, 0x81, 0xb8, 0x85, 0x60, 0x2e, 0x2b, 0x88, 0x9b, 0xc4, 0x06, 0x36, 0xdd, 0x18,
	0x10, 0x00, 0x00, 0xff, 0xff, 0x9d, 0x8e, 0x1b, 0xe2, 0x72, 0x00, 0x00, 0x00,
}
