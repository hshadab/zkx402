#!/usr/bin/perl
#
# ====================================================================
# Written by Andy Polyakov, @dot-asm, initially for use with OpenSSL.
# ====================================================================
#
# ChaCha20 for RISC-V.
#
# March 2019.
#
# This is transliteration of MIPS module [without big-endian option].
#
# 14.1 cycles per byte on U74 for aligned input, ~70% faster than
# compiler-generated code. Misaligned input is processed in 16.4 cpb.
# C910 processes one byte in 13.7 cycles. JH7110 (U74+zbb) - in 10.3.
#
# October 2023.
#
# Add a "teaser" vector implementation. It's a "teaser," because one
# can improve it further for longer inputs. But it makes no sense to
# invest time prior vector-capable hardware appears, so that one can
# make suitable choices. For now it's just a qemu exercise...
#
# June 2024.
#
# Add CHERI support.
#
######################################################################
#
($zero,$ra,$sp,$gp,$tp) = map("x$_",(0..4));
($a0,$a1,$a2,$a3,$a4,$a5,$a6,$a7)=map("x$_",(10..17));
($s0,$s1,$s2,$s3,$s4,$s5,$s6,$s7,$s8,$s9,$s10,$s11)=map("x$_",(8,9,18..27));
($t0,$t1,$t2,$t3,$t4,$t5,$t6)=map("x$_",(5..7, 28..31));
#
######################################################################

$flavour = shift;	# "cheri" is the only meaningful option

my @x = map("x$_",(16..31));
my @y = map("x$_",(5..9,13,14));
my $at = @y[-1];
my ($out, $inp, $len, $key, $counter) = ($a0,$a1,$a2,$a3,$a4);

sub ROUND {
my ($a0,$b0,$c0,$d0)=@_;
my ($a1,$b1,$c1,$d1)=map(($_&~3)+(($_+1)&3),($a0,$b0,$c0,$d0));
my ($a2,$b2,$c2,$d2)=map(($_&~3)+(($_+1)&3),($a1,$b1,$c1,$d1));
my ($a3,$b3,$c3,$d3)=map(($_&~3)+(($_+1)&3),($a2,$b2,$c2,$d2));

$code.=<<___;
	add		@x[$a0],@x[$a0],@x[$b0]		# Q0
	 add		@x[$a1],@x[$a1],@x[$b1]		# Q1
	  add		@x[$a2],@x[$a2],@x[$b2]		# Q2
	   add		@x[$a3],@x[$a3],@x[$b3]		# Q3
	xor		@x[$d0],@x[$d0],@x[$a0]
	 xor		@x[$d1],@x[$d1],@x[$a1]
	  xor		@x[$d2],@x[$d2],@x[$a2]
	   xor		@x[$d3],@x[$d3],@x[$a3]
#ifdef	__riscv_zbb
	rorw		@x[$d0],@x[$d0],16
	 rorw		@x[$d1],@x[$d1],16
	  rorw		@x[$d2],@x[$d2],16
	   rorw		@x[$d3],@x[$d3],16
#else
	srlw		@y[0],@x[$d0],16
	 srlw		@y[1],@x[$d1],16
	  srlw		@y[2],@x[$d2],16
	   srlw		@y[3],@x[$d3],16
	sll		@x[$d0],@x[$d0],16
	 sll		@x[$d1],@x[$d1],16
	  sll		@x[$d2],@x[$d2],16
	   sll		@x[$d3],@x[$d3],16
	or		@x[$d0],@x[$d0],@y[0]
	 or		@x[$d1],@x[$d1],@y[1]
	  or		@x[$d2],@x[$d2],@y[2]
	   or		@x[$d3],@x[$d3],@y[3]
#endif

	add		@x[$c0],@x[$c0],@x[$d0]
	 add		@x[$c1],@x[$c1],@x[$d1]
	  add		@x[$c2],@x[$c2],@x[$d2]
	   add		@x[$c3],@x[$c3],@x[$d3]
	xor		@x[$b0],@x[$b0],@x[$c0]
	 xor		@x[$b1],@x[$b1],@x[$c1]
	  xor		@x[$b2],@x[$b2],@x[$c2]
	   xor		@x[$b3],@x[$b3],@x[$c3]
#ifdef	__riscv_zbb
	rorw		@x[$b0],@x[$b0],20
	 rorw		@x[$b1],@x[$b1],20
	  rorw		@x[$b2],@x[$b2],20
	   rorw		@x[$b3],@x[$b3],20
#else
	srlw		@y[0],@x[$b0],20
	 srlw		@y[1],@x[$b1],20
	  srlw		@y[2],@x[$b2],20
	   srlw		@y[3],@x[$b3],20
	sll		@x[$b0],@x[$b0],12
	 sll		@x[$b1],@x[$b1],12
	  sll		@x[$b2],@x[$b2],12
	   sll		@x[$b3],@x[$b3],12
	or		@x[$b0],@x[$b0],@y[0]
	 or		@x[$b1],@x[$b1],@y[1]
	  or		@x[$b2],@x[$b2],@y[2]
	   or		@x[$b3],@x[$b3],@y[3]
#endif

	add		@x[$a0],@x[$a0],@x[$b0]
	 add		@x[$a1],@x[$a1],@x[$b1]
	  add		@x[$a2],@x[$a2],@x[$b2]
	   add		@x[$a3],@x[$a3],@x[$b3]
	xor		@x[$d0],@x[$d0],@x[$a0]
	 xor		@x[$d1],@x[$d1],@x[$a1]
	  xor		@x[$d2],@x[$d2],@x[$a2]
	   xor		@x[$d3],@x[$d3],@x[$a3]
#ifdef	__riscv_zbb
	rorw		@x[$d0],@x[$d0],24
	 rorw		@x[$d1],@x[$d1],24
	  rorw		@x[$d2],@x[$d2],24
	   rorw		@x[$d3],@x[$d3],24
#else
	srlw		@y[0],@x[$d0],24
	 srlw		@y[1],@x[$d1],24
	  srlw		@y[2],@x[$d2],24
	   srlw		@y[3],@x[$d3],24
	sll		@x[$d0],@x[$d0],8
	 sll		@x[$d1],@x[$d1],8
	  sll		@x[$d2],@x[$d2],8
	   sll		@x[$d3],@x[$d3],8
	or		@x[$d0],@x[$d0],@y[0]
	 or		@x[$d1],@x[$d1],@y[1]
	  or		@x[$d2],@x[$d2],@y[2]
	   or		@x[$d3],@x[$d3],@y[3]
#endif

	add		@x[$c0],@x[$c0],@x[$d0]
	 add		@x[$c1],@x[$c1],@x[$d1]
	  add		@x[$c2],@x[$c2],@x[$d2]
	   add		@x[$c3],@x[$c3],@x[$d3]
	xor		@x[$b0],@x[$b0],@x[$c0]
	 xor		@x[$b1],@x[$b1],@x[$c1]
	  xor		@x[$b2],@x[$b2],@x[$c2]
	   xor		@x[$b3],@x[$b3],@x[$c3]
#ifdef	__riscv_zbb
	rorw		@x[$b0],@x[$b0],25
	 rorw		@x[$b1],@x[$b1],25
	  rorw		@x[$b2],@x[$b2],25
	   rorw		@x[$b3],@x[$b3],25
#else
	srlw		@y[0],@x[$b0],25
	 srlw		@y[1],@x[$b1],25
	  srlw		@y[2],@x[$b2],25
	   srlw		@y[3],@x[$b3],25
	sll		@x[$b0],@x[$b0],7
	 sll		@x[$b1],@x[$b1],7
	  sll		@x[$b2],@x[$b2],7
	   sll		@x[$b3],@x[$b3],7
	or		@x[$b0],@x[$b0],@y[0]
	 or		@x[$b1],@x[$b1],@y[1]
	  or		@x[$b2],@x[$b2],@y[2]
	   or		@x[$b3],@x[$b3],@y[3]
#endif
___
}

$code.=<<___;
#if __riscv_xlen == 32
# if __SIZEOF_POINTER__ == 8
#  define PUSH	csc
#  define POP	clc
# else
#  define PUSH	sw
#  define POP	lw
# endif
# define srlw	srl
# define rorw	ror
#elif __riscv_xlen == 64
# if __SIZEOF_POINTER__ == 16
#  define PUSH	csc
#  define POP	clc
# else
#  define PUSH	sd
#  define POP	ld
# endif
#else
# error "unsupported __riscv_xlen"
#endif
#define FRAMESIZE		(64+16*__SIZEOF_POINTER__)

.text
.option	pic

.type	__ChaCha,\@function
__ChaCha:
	lw		@x[0], 4*0($sp)
	lw		@x[1], 4*1($sp)
	lw		@x[2], 4*2($sp)
	lw		@x[3], 4*3($sp)
	lw		@x[4], 4*4($sp)
	lw		@x[5], 4*5($sp)
	lw		@x[6], 4*6($sp)
	lw		@x[7], 4*7($sp)
	lw		@x[8], 4*8($sp)
	lw		@x[9], 4*9($sp)
	lw		@x[10],4*10($sp)
	lw		@x[11],4*11($sp)
	mv		@x[12],$a5
	lw		@x[13],4*13($sp)
	lw		@x[14],4*14($sp)
	lw		@x[15],4*15($sp)
.Loop:
	addi		$at,$at,-1
___
	&ROUND(0, 4, 8, 12);
	&ROUND(0, 5, 10, 15);
$code.=<<___;
	bnez		$at,.Loop

	lw		@y[0], 4*0($sp)
	lw		@y[1], 4*1($sp)
	lw		@y[2], 4*2($sp)
	lw		@y[3], 4*3($sp)
	add		@x[0],@x[0],@y[0]
	lw		@y[0],4*4($sp)
	add		@x[1],@x[1],@y[1]
	lw		@y[1],4*5($sp)
	add		@x[2],@x[2],@y[2]
	lw		@y[2],4*6($sp)
	add		@x[3],@x[3],@y[3]
	lw		@y[3],4*7($sp)
	add		@x[4],@x[4],@y[0]
	lw		@y[0],4*8($sp)
	add		@x[5],@x[5],@y[1]
	lw		@y[1], 4*9($sp)
	add		@x[6],@x[6],@y[2]
	lw		@y[2],4*10($sp)
	add		@x[7],@x[7],@y[3]
	lw		@y[3],4*11($sp)
	add		@x[8],@x[8],@y[0]
	#lw		@y[0],4*12($sp)
	add		@x[9],@x[9],@y[1]
	lw		@y[1],4*13($sp)
	add		@x[10],@x[10],@y[2]
	lw		@y[2],4*14($sp)
	add		@x[11],@x[11],@y[3]
	lw		@y[3],4*15($sp)
	add		@x[12],@x[12],$a5
	add		@x[13],@x[13],@y[1]
	add		@x[14],@x[14],@y[2]
	add		@x[15],@x[15],@y[3]
	ret
.size	__ChaCha,.-__ChaCha

.globl	ChaCha20_ctr32
.type	ChaCha20_ctr32,\@function
ChaCha20_ctr32:
	caddi		$sp,$sp,-FRAMESIZE
	PUSH		$ra, (FRAMESIZE-1*__SIZEOF_POINTER__)($sp)
	PUSH		$s0, (FRAMESIZE-2*__SIZEOF_POINTER__)($sp)
	PUSH		$s1, (FRAMESIZE-3*__SIZEOF_POINTER__)($sp)
	PUSH		$s2, (FRAMESIZE-4*__SIZEOF_POINTER__)($sp)
	PUSH		$s3, (FRAMESIZE-5*__SIZEOF_POINTER__)($sp)
	PUSH		$s4, (FRAMESIZE-6*__SIZEOF_POINTER__)($sp)
	PUSH		$s5, (FRAMESIZE-7*__SIZEOF_POINTER__)($sp)
	PUSH		$s6, (FRAMESIZE-8*__SIZEOF_POINTER__)($sp)
	PUSH		$s7, (FRAMESIZE-9*__SIZEOF_POINTER__)($sp)
	PUSH		$s8, (FRAMESIZE-10*__SIZEOF_POINTER__)($sp)
	PUSH		$s9, (FRAMESIZE-11*__SIZEOF_POINTER__)($sp)
	PUSH		$s10,(FRAMESIZE-12*__SIZEOF_POINTER__)($sp)
	PUSH		$s11,(FRAMESIZE-13*__SIZEOF_POINTER__)($sp)

	lui		@x[0],0x61707+1		# compose sigma
	lui		@x[1],0x33206
	lui		@x[2],0x79622+1
	lui		@x[3],0x6b206
	addi		@x[0],@x[0],-0x79b
	addi		@x[1],@x[1],0x46e
	addi		@x[2],@x[2],-0x2ce
	addi		@x[3],@x[3],0x574

	lw		@x[4], 4*0($key)
	lw		@x[5], 4*1($key)
	lw		@x[6], 4*2($key)
	lw		@x[7], 4*3($key)
	lw		@x[8], 4*4($key)
	lw		@x[9], 4*5($key)
	lw		@x[10],4*6($key)
	lw		@x[11],4*7($key)

	lw		@x[12],4*0($counter)
	lw		@x[13],4*1($counter)
	lw		@x[14],4*2($counter)
	lw		@x[15],4*3($counter)

	sw		@x[0], 4*0($sp)
	sw		@x[1], 4*1($sp)
	sw		@x[2], 4*2($sp)
	sw		@x[3], 4*3($sp)
	sw		@x[4], 4*4($sp)
	sw		@x[5], 4*5($sp)
	sw		@x[6], 4*6($sp)
	sw		@x[7], 4*7($sp)
	sw		@x[8], 4*8($sp)
	sw		@x[9], 4*9($sp)
	sw		@x[10],4*10($sp)
	sw		@x[11],4*11($sp)
	mv		$a5,@x[12]
	sw		@x[13],4*13($sp)
	sw		@x[14],4*14($sp)
	sw		@x[15],4*15($sp)

	li		$at,10
	jal		.Loop

	sltiu		$at,$len,64
	or		$ra,$inp,$out
	andi		$ra,$ra,3		# both are aligned?
	bnez		$at,.Ltail

	beqz		$ra,.Loop_aligned

.Loop_misaligned:
	lb		@y[0],0($inp)
	lb		@y[1],1($inp)
	srl		@y[4],@x[0],8
	lb		@y[2],2($inp)
	srl		@y[5],@x[0],16
	lb		@y[3],3($inp)
	srl		@y[6],@x[0],24
___
for(my $i=0; $i<15; $i++) {
my $j=4*$i;
my $k=4*($i+1);
$code.=<<___;
	xor		@x[$i],@x[$i],@y[0]
	lb		@y[0],$k+0($inp)
	xor		@y[4],@y[4],@y[1]
	lb		@y[1],$k+1($inp)
	xor		@y[5],@y[5],@y[2]
	lb		@y[2],$k+2($inp)
	xor		@y[6],@y[6],@y[3]
	lb		@y[3],$k+3($inp)
	sb		@x[$i],$j+0($out)
	sb		@y[4],$j+1($out)
	srl		@y[4],@x[$i+1],8
	sb		@y[5],$j+2($out)
	srl		@y[5],@x[$i+1],16
	sb		@y[6],$j+3($out)
	srl		@y[6],@x[$i+1],24
___
}
$code.=<<___;
	xor		@x[15],@x[15],@y[0]
	xor		@y[4],@y[4],@y[1]
	xor		@y[5],@y[5],@y[2]
	xor		@y[6],@y[6],@y[3]
	sb		@x[15],60($out)
	addi		$a5,$a5,1		# next counter value
	sb		@y[4],61($out)
	addi		$len,$len,-64
	sb		@y[5],62($out)
	caddi		$inp,$inp,64
	sb		@y[6],63($out)
	caddi		$out,$out,64
	beqz		$len,.Ldone

	sltiu		@y[4],$len,64
	li		$at,10
	jal		__ChaCha

	beqz		@y[4],.Loop_misaligned

	j		.Ltail

.Loop_aligned:
	lw		@y[0],0($inp)
	lw		@y[1],4($inp)
	lw		@y[2],8($inp)
	lw		@y[3],12($inp)
___
for (my $i=0; $i<12; $i+=4) {
my $j = 4*$i;
my $k = 4*($i+4);
$code.=<<___;
	xor		@x[$i+0],@x[$i+0],@y[0]
	lw		@y[0],$k+0($inp)
	xor		@x[$i+1],@x[$i+1],@y[1]
	lw		@y[1],$k+4($inp)
	xor		@x[$i+2],@x[$i+2],@y[2]
	lw		@y[2],$k+8($inp)
	xor		@x[$i+3],@x[$i+3],@y[3]
	lw		@y[3],$k+12($inp)
	sw		@x[$i+0],$j+0($out)
	sw		@x[$i+1],$j+4($out)
	sw		@x[$i+2],$j+8($out)
	sw		@x[$i+3],$j+12($out)
___
}
$code.=<<___;
	xor		@x[12],@x[12],@y[0]
	xor		@x[13],@x[13],@y[1]
	xor		@x[14],@x[14],@y[2]
	xor		@x[15],@x[15],@y[3]
	sw		@x[12],48($out)
	addi		$a5,$a5,1		# next counter value
	sw		@x[13],52($out)
	addi		$len,$len,-64
	sw		@x[14],56($out)
	caddi		$inp,$inp,64
	sw		@x[15],60($out)
	caddi		$out,$out,64
	sltiu		@y[4],$len,64
	beqz		$len,.Ldone

	li		$at,10
	jal		__ChaCha

	beqz		@y[4],.Loop_aligned

.Ltail:
	cmove		$ra,$sp
	sw		@x[1], 4*1($sp)
	sw		@x[2], 4*2($sp)
	sw		@x[3], 4*3($sp)
	sw		@x[4], 4*4($sp)
	sw		@x[5], 4*5($sp)
	sw		@x[6], 4*6($sp)
	sw		@x[7], 4*7($sp)
	sw		@x[8], 4*8($sp)
	sw		@x[9], 4*9($sp)
	sw		@x[10],4*10($sp)
	sw		@x[11],4*11($sp)
	sw		@x[12],4*12($sp)
	sw		@x[13],4*13($sp)
	sw		@x[14],4*14($sp)
	sw		@x[15],4*15($sp)

.Loop_tail:
	sltiu		$at,$len,4
	bnez		$at,.Last_word

	caddi		$ra,$ra,4
	lb		@y[0],0($inp)
	lb		@y[1],1($inp)
	lb		@y[2],2($inp)
	addi		$len,$len,-4
	lb		@y[3],3($inp)
	caddi		$inp,$inp,4
	xor		@y[0],@y[0],@x[0]
	srl		@x[0],@x[0],8
	xor		@y[1],@y[1],@x[0]
	srl		@x[0],@x[0],8
	xor		@y[2],@y[2],@x[0]
	srl		@x[0],@x[0],8
	xor		@y[3],@y[3],@x[0]
	lw		@x[0],0($ra)
	sb		@y[0],0($out)
	sb		@y[1],1($out)
	sb		@y[2],2($out)
	sb		@y[3],3($out)
	caddi		$out,$out,4
	j		.Loop_tail

.Last_word:
	beqz		$len,.Ldone
	addi		$len,$len,-1
	lb		@y[0],0($inp)
	caddi		$inp,$inp,1
	xor		@y[0],@y[0],@x[0]
	srl		@x[0],@x[0],8
	sb		@y[0],0($out)
	caddi		$out,$out,1
	j		.Last_word

.Ldone:
	POP		$ra, (FRAMESIZE-1*__SIZEOF_POINTER__)($sp)
	POP		$s0, (FRAMESIZE-2*__SIZEOF_POINTER__)($sp)
	POP		$s1, (FRAMESIZE-3*__SIZEOF_POINTER__)($sp)
	POP		$s2, (FRAMESIZE-4*__SIZEOF_POINTER__)($sp)
	POP		$s3, (FRAMESIZE-5*__SIZEOF_POINTER__)($sp)
	POP		$s4, (FRAMESIZE-6*__SIZEOF_POINTER__)($sp)
	POP		$s5, (FRAMESIZE-7*__SIZEOF_POINTER__)($sp)
	POP		$s6, (FRAMESIZE-8*__SIZEOF_POINTER__)($sp)
	POP		$s7, (FRAMESIZE-9*__SIZEOF_POINTER__)($sp)
	POP		$s8, (FRAMESIZE-10*__SIZEOF_POINTER__)($sp)
	POP		$s9, (FRAMESIZE-11*__SIZEOF_POINTER__)($sp)
	POP		$s10,(FRAMESIZE-12*__SIZEOF_POINTER__)($sp)
	POP		$s11,(FRAMESIZE-13*__SIZEOF_POINTER__)($sp)
	caddi		$sp,$sp,FRAMESIZE
	ret
.size	ChaCha20_ctr32,.-ChaCha20_ctr32
___

sub HROUND {
my ($a, $b, $c, $d, $t) = @_;

$code.=<<___
	vadd.vv		$a, $a, $b	# a += b
	vxor.vv		$d, $d, $a	# d ^= a
	vsrl.vi		$t, $d, 16
	vsll.vi		$d, $d, 16
	vor.vv		$d, $d, $t	# d >>>= 16

	vadd.vv		$c, $c, $d	# c += d
	vxor.vv		$b, $b, $c	# b ^= c
	vsrl.vi		$t, $b, 20
	vsll.vi		$b, $b, 12
	vor.vv		$b, $b, $t	# b >>>= 20

	vadd.vv		$a, $a, $b	# a += b
	vxor.vv		$d, $d, $a	# d ^= a
	vsrl.vi		$t, $d, 24
	vsll.vi		$d, $d, 8
	vor.vv		$d, $d, $t	# d >>>= 24

	vadd.vv		$c, $c, $d	# c += d
	vxor.vv		$b, $b, $c	# b ^= c
	vsrl.vi		$t, $b, 25
	vsll.vi		$b, $b, 7
	vor.vv		$b, $b, $t	# b >>>= 25
___
}

$code.=<<___;
#if defined(__riscv_v) && __riscv_v >= 1000000
.globl	ChaCha20_ctr32_v
.type	ChaCha20_ctr32_v,\@function
ChaCha20_ctr32_v:
	lla		$t0, sigma
	vsetivli	$zero, 4, e32

	vle32.v		v8, ($t0)	# a'
	addi		$t1, $key, 16
	vle32.v		v9, ($key)	# b'
	vle32.v		v10, ($t1)	# c'
	vle32.v		v11, ($counter)	# d'

	addi		$t1, $t0, 16
	addi		$t2, $t0, 32
	addi		$t3, $t0, 48
	vle32.v		v12, ($t1)
	vle32.v		v13, ($t2)
	vle32.v		v14, ($t3)
	vmv.x.s		$counter, v11	# put aside the 32-bit counter value
	li		$t0, 64
	j		.Loop_enter_v

.Loop_outer_v:
	addi		$counter, $counter, 1
	vsetivli	zero, 4, e32
	vmv.s.x		v11, $counter	# d'
.Loop_enter_v:
	vmv.v.v		v0, v8		# a = a'
	vmv.v.v		v1, v9		# b = b'
	vmv.v.v		v2, v10		# c = c'
	vmv.v.v		v3, v11		# d = d'
	li		$a5, 10
.Loop_v:
___
	&HROUND(map("v$_", (0..3,4)));
$code.=<<___;
	vrgather.vv	v4, v1, v12	# b >>>= 32
	vrgather.vv	v5, v2, v13	# c >>>= 64
	vrgather.vv	v6, v3, v14	# d >>>= 96
___
	&HROUND(map("v$_", (0,4..6,1)));
$code.=<<___;
	vrgather.vv	v1, v4, v14	# b >>>= 96
	vrgather.vv	v2, v5, v13	# c >>>= 64
	vrgather.vv	v3, v6, v12	# d >>>= 32

	addi		$a5, $a5, -1
	bnez		$a5, .Loop_v

	vadd.vv		v0, v0, v8	# a += a'
	vadd.vv		v1, v1, v9	# b += b'
	vadd.vv		v2, v2, v10	# c += c'
	vadd.vv		v3, v3, v11	# d += d'

	vsetivli	zero, 16, e8
	bltu		$len, $t0, .Ltail_v

	caddi		$t1, $inp, 16
	caddi		$t2, $inp, 32
	caddi		$t3, $inp, 48
	vle8.v		v4, ($inp)
	caddi		$inp, $inp, 64
	addi		$len, $len, -64
	vle8.v		v5, ($t1)
	caddi		$t1, $out, 16
	vle8.v		v6, ($t2)
	caddi		$t2, $out, 32
	vle8.v		v7, ($t3)
	caddi		$t3, $out, 48
	vxor.vv		v4, v4, v0
	vxor.vv		v5, v5, v1
	vxor.vv		v6, v6, v2
	vxor.vv		v7, v7, v3
	vse8.v		v4, ($out)
	caddi		$out, $out, 64
	vse8.v		v5, ($t1)
	vse8.v		v6, ($t2)
	vse8.v		v7, ($t3)
	bnez		$len, .Loop_outer_v

	ret

.Ltail_v:
	li		$t0, 16
	bleu		$len, $t0, .Last_v

	vle8.v		v4, ($inp)
	caddi		$inp, $inp, 16
	addi		$len, $len, -16
	vxor.vv		v4, v4, v0
	vmv.v.v		v0, v1
	vse8.v		v4, ($out)
	caddi		$out, $out, 16
	bleu		$len, $t0, .Last_v

	vle8.v		v5, ($inp)
	caddi		$inp, $inp, 16
	addi		$len, $len, -16
	vxor.vv		v5, v5, v1
	vmv.v.v		v0, v2
	vse8.v		v5, ($out)
	caddi		$out, $out, 16
	bleu		$len, $t0, .Last_v

	vle8.v		v6, ($inp)
	caddi		$inp, $inp, 16
	addi		$len, $len, -16
	vxor.vv		v6, v6, v2
	vmv.v.v		v0, v3
	vse8.v		v6, ($out)
	caddi	$out, $out, 16

.Last_v:
	vsetvli		zero, $len, e8
	vle8.v		v4, ($inp)
	vxor.vv		v4, v4, v0
	vse8.v		v4, ($out)

	ret
.size	ChaCha20_ctr32_v,.-ChaCha20_ctr32_v
#endif

.section	.rodata
.align	4
sigma:
.word	0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
.word	1,2,3,0,    2,3,0,1,    3,0,1,2,    0,0,0,0
.string	"ChaCha20 for RISC-V, CRYPTOGAMS by \@dot-asm"
___

foreach (split("\n", $code)) {
    if ($flavour =~ "cheri") {
	s/\(x([0-9]+)\)/(c$1)/ and s/\b([ls][bhwd]u?)\b/c$1/;
	s/\b(PUSH|POP)(\s+)x([0-9]+)/$1$2c$3/ or
	s/\b(ret|jal)\b/c$1/;
	s/\bcaddi?\b/cincoffset/ and s/\bx([0-9]+,)/c$1/g or
	m/\bcmove\b/ and s/\bx([0-9]+)/c$1/g;
    } else {
	s/\bcaddi?\b/add/ or
	s/\bcmove\b/mv/;
    }
    print $_, "\n";
}

close STDOUT;
