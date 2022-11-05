
import idaapi
import ida_hexrays


AMX_NONE = 0
AMX_OP0 = 1
AMX_OP1 = 2
AMX_OP2 = 3
AMX_OP3 = 4
AMX_OP4 = 5
AMX_OP5 = 6
AMX_OP6 = 7
AMX_OP7 = 8
AMX_OP8 = 9
AMX_OP9 = 10
AMX_OP10 = 11
AMX_OP11 = 12
AMX_OP12 = 13
AMX_OP13 = 14
AMX_OP14 = 15
AMX_OP15 = 16
AMX_OP16 = 17
AMX_OP17 = 18
AMX_OP18 = 19
AMX_OP19 = 20
AMX_OP20 = 21
AMX_OP21 = 22
AMX_OP22 = 23

OP_NAMES = {
	AMX_OP0: "AMXLDX",
	AMX_OP1: "AMXLDY",
	AMX_OP2: "AMXSTX",
	AMX_OP3: "AMXSTY",
	AMX_OP4: "AMXLDZ",
	AMX_OP5: "AMXSTZ",
	AMX_OP6: "AMXLDZI",
	AMX_OP7: "AMXSTZI",
	AMX_OP8: "AMXEXTRX", # amxextrx?
	AMX_OP9: "AMXEXTRY", # amxextry?
	AMX_OP10: "AMXFMA64",
	AMX_OP11: "AMXFMS64",
	AMX_OP12: "AMXFMA32",
	AMX_OP13: "AMXFMS32",
	AMX_OP14: "AMXMAC16",
	AMX_OP15: "AMXFMA16",
	AMX_OP16: "AMXFMS16",
	AMX_OP17: "AMX17", # amxset / amxclr
	AMX_OP18: "AMXVECINT",
	AMX_OP19: "AMXVECFP",
	AMX_OP20: "AMXMATINT",
	AMX_OP21: "AMXMATFP",
	AMX_OP22: "AMXGENLUT",
}

OP_INTRINSIC_NAMES = {
	AMX_OP0:  "__amx_ldx",
	AMX_OP1:  "__amx_ldy",
	AMX_OP2:  "__amx_stx",
	AMX_OP3:  "__amx_sty",
	AMX_OP4:  "__amx_ldz",
	AMX_OP5:  "__amx_stz",
	AMX_OP6:  "__amx_ldzi",
	AMX_OP7:  "__amx_stzi",
	AMX_OP8:  "__amx_extrx",
	AMX_OP9:  "__amx_extry",
	AMX_OP10: "__amx_fma64",
	AMX_OP11: "__amx_fms64",
	AMX_OP12: "__amx_fma32",
	AMX_OP13: "__amx_fms32",
	AMX_OP14: "__amx_mac16",
	AMX_OP15: "__amx_fma16",
	AMX_OP16: "__amx_fms16",
	AMX_OP17: "__amx_op17", # amxset / amxclr
	AMX_OP18: "__amx_vecint",
	AMX_OP19: "__amx_vecfp",
	AMX_OP20: "__amx_matint",
	AMX_OP21: "__amx_matfp",
	AMX_OP22: "__amx_genlut",
}

def decode_AMX(d, insn):
	if (d & 0xfffffC00) == 0x00201000:
		Xr = d & 31
		m = (d >> 5) & 31
		if m <= AMX_OP22 - AMX_OP0:
			#insn.itype = idaapi.ARM_nop
			insn.itype = idaapi.ARM_hlt
			insn.segpref = 14
			if m == 17:
				insn.Op1.type = idaapi.o_imm
				insn.Op1.value = Xr
				insn.Op1.dtype = idaapi.dt_byte
			else:
				insn.Op1.type = idaapi.o_reg
				insn.Op1.reg = Xr + 129
				insn.Op1.dtype = idaapi.dt_qword
			insn.insnpref = AMX_OP0 + m
			insn.size = 4
		return True
	return False

class Aarch64AMXHook(idaapi.IDP_Hooks):
	CUSTOM_INSTRUCTIONS = {idaapi.ARM_hlt}
	INDENT = 16
	def ev_ana_insn(self, outctx):
		return outctx.size if decode_AMX(idaapi.get_dword(outctx.ea), outctx) else 0

	def ev_emu_insn(self, insn):
		if insn.itype != idaapi.ARM_brk:
			return False
		return True

	def ev_out_mnem(self, outctx):
		if outctx.insn.itype in self.CUSTOM_INSTRUCTIONS:
			mnem = OP_NAMES.get(ord(outctx.insn.insnpref), None)
			if mnem is not None:
				outctx.out_custom_mnem(mnem, self.INDENT)
				return 1
		return 0

class MicroInstruction(ida_hexrays.minsn_t):

	def __init__(self, opcode, ea):
		ida_hexrays.minsn_t.__init__(self, ea)
		self.opcode = opcode
		self.l.zero()
		self.r.zero()
		self.d.zero()

class CallBuilder():

	def __init__(self, cdg, name, return_type=idaapi.tinfo_t(idaapi.BT_VOID)):
		self.emitted = False
		self.cdg = cdg
		self.callinfo = ida_hexrays.mcallinfo_t()
		self.callinfo.callee = idaapi.BADADDR
		self.callinfo.solid_args = 0
		self.callinfo.call_spd = 0
		self.callinfo.stkargs_top = 0
		self.callinfo.cc = idaapi.CM_CC_FASTCALL
		self.callinfo.return_type = return_type
		self.callinfo.flags = idaapi.FCI_SPLOK | idaapi.FCI_FINAL | idaapi.FCI_PROP
		self.callinfo.role = idaapi.ROLE_UNK

		glbhigh_off = cdg.mba.get_stack_region().off + cdg.mba.get_stack_region().size
		# what memory is visible to the call : GLBLOW - GLBHIGH
		self.callinfo.visible_memory.add(ida_hexrays.ivl_t(0x00, 0x100000))
		self.callinfo.visible_memory.add(ida_hexrays.ivl_t(glbhigh_off, 0xFFFFFFFFFFFFFFFF - glbhigh_off))
		# spoiled locations : GLBLOW - GLBHIGH
		self.callinfo.spoiled.mem.add(ida_hexrays.ivl_t(0x00, 0x100000))
		self.callinfo.spoiled.mem.add(ida_hexrays.ivl_t(glbhigh_off, 0xFFFFFFFFFFFFFFFF - glbhigh_off))

		self.callins = MicroInstruction(ida_hexrays.m_call, self.cdg.insn.ea)
		self.callins.l.make_helper(name)
		self.callins.d.t = ida_hexrays.mop_f
		self.callins.d.size = 0
		self.callins.d.f = self.callinfo

		if (return_type.is_void()):
			self.ins = self.callins
		else:
			self.callins.d.size = return_type.get_size()
			self.ins = MicroInstruction(ida_hexrays.m_mov, self.cdg.insn.ea)
			self.ins.l.t = ida_hexrays.mop_d
			self.ins.l.d = self.callins
			self.ins.l.size = self.callins.d.size
			self.ins.d.t = ida_hexrays.mop_r
			self.ins.d.r = 0x00
			self.ins.d.size = self.callins.d.size

	def add_register_argument(self, t, operand):
		ca = ida_hexrays.mcallarg_t()
		ca.t = idaapi.mop_r
		ca.r = operand
		ca.type = t
		ca.size = t.get_size()
		self.callinfo.args.push_back(ca)
		self.callinfo.solid_args += 1

	def set_return_register(self, reg):
		self.ins.d.r = reg

	def emit(self):
		if self.emitted == False:
			self.cdg.mb.insert_into_block(self.ins, self.cdg.mb.tail)
			self.emitted = True

class AMXFilter(ida_hexrays.microcode_filter_t):
	def __init__(self):
		ida_hexrays.microcode_filter_t.__init__(self)
		ida_hexrays.install_microcode_filter(self, True)

	def match(self, cdg):
		return cdg.insn.itype == idaapi.ARM_hlt and cdg.insn.insnpref != AMX_NONE

	def apply(self, cdg):
		op = ord(cdg.insn.insnpref)
		intrinsic_name = OP_INTRINSIC_NAMES.get(op, '__amx%d' % op)
		if cdg.insn.Op1.type == idaapi.o_reg:
			builder = CallBuilder(cdg, intrinsic_name)
			builder.add_register_argument(idaapi.tinfo_t(idaapi.BT_INT64 | idaapi.BTMT_UNSIGNED), cdg.load_operand(0))
			builder.emit()
		elif cdg.insn.Op1.type == idaapi.o_imm:
			if op == AMX_OP17 and cdg.insn.Op1.value == 0:
				builder = CallBuilder(cdg, '__amx_begin')
				builder.emit()
			elif op == AMX_OP17 and cdg.insn.Op1.value == 1:
				builder = CallBuilder(cdg, '__amx_end')
				builder.emit()
			else:
				builder = CallBuilder(cdg, '%s_%d' % (intrinsic_name, cdg.insn.Op1.value))
				builder.emit()

		return idaapi.MERR_OK


class Aarch64AMXPlugin(idaapi.plugin_t):
	flags = idaapi.PLUGIN_PROC | idaapi.PLUGIN_HIDE
	comment = "Aarch64 Apple AMX extension"
	wanted_hotkey = ""
	help = "Runs transparently"
	wanted_name = "Aarch64 AMX"
	hook = None
	enabled = 1

	def init(self):
		if idaapi.ph_get_id() != idaapi.PLFM_ARM or idaapi.BADADDR <= 0xFFFFFFFF:
			return idaapi.PLUGIN_SKIP
		if not ida_hexrays.init_hexrays_plugin():
			print("[-] {0} : no decompiler available, skipping".format(self.wanted_name))
			return idaapi.PLUGIN_SKIP
		print("%s init"%self.comment)
		self.hook = Aarch64AMXHook()
		self.hook.hook()
		self.filter = AMXFilter()
		return idaapi.PLUGIN_KEEP

	def run():
		pass

	def term(self):
		if self.hook is not None:
			self.hook.unhook()
		print("%s unloaded"%self.comment)

def PLUGIN_ENTRY():
	return Aarch64AMXPlugin()
