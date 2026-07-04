/* Serial Ghidra postScript for LLM4Decompile benchmark baseline.
 *
 * Single-threaded decompilation compatible with Ghidra 12+ (Java postScript).
 * Output format matches service/app/preprocessor.py (// Function: markers).
 */

import java.io.FileWriter;
import java.io.PrintWriter;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.decompiler.DecompiledFunction;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.CodeUnit;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.Instruction;
import ghidra.util.Msg;

public class DecompileSerial extends GhidraScript {

	@Override
	protected void run() throws Exception {
		String[] args = getScriptArgs();
		if (args.length < 1 || args[0].trim().isEmpty()) {
			throw new IllegalArgumentException("Usage: DecompileSerial.java <output_path>");
		}

		String outputPath = args[0].trim();
		int totalFunctions = currentProgram.getFunctionManager().getFunctionCount();
		monitor.initialize(totalFunctions);

		DecompileOptions options = new DecompileOptions();
		DecompInterface decompiler = new DecompInterface();
		decompiler.setOptions(options);
		decompiler.openProgram(currentProgram);
		decompiler.toggleSyntaxTree(false);

		long startMs = System.currentTimeMillis();
		int written = 0;
		try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
			FunctionIterator iterator = currentProgram.getFunctionManager().getFunctions(true);
			while (iterator.hasNext()) {
				monitor.checkCanceled();
				Function function = iterator.next();
				writeFunction(writer, function, decompiler, options);
				written++;
			}
			writer.flush();
		}
		finally {
			decompiler.dispose();
		}

		long elapsedMs = System.currentTimeMillis() - startMs;
		println("DECOMPILE_SERIAL functions=" + totalFunctions + " written=" + written);
		println("DECOMPILE_SERIAL timing_ms decompile=" + elapsedMs);
	}

	private void writeFunction(
		PrintWriter writer,
		Function function,
		DecompInterface decompiler,
		DecompileOptions options
	) throws Exception {
		Address entryPoint = function.getEntryPoint();
		String marker = "// Function: " + function.toString();
		writer.write(marker);

		CodeUnit codeUnitAt = function.getProgram().getListing().getCodeUnitAt(entryPoint);
		if (codeUnitAt == null || !(codeUnitAt instanceof Instruction)) {
			writer.write(function.getPrototypeString(false, false));
			writer.write(';');
			return;
		}

		monitor.setMessage("Decompiling " + function.getName());
		DecompileResults results = decompiler.decompileFunction(function, options.getDefaultTimeout(), monitor);
		String errorMessage = results.getErrorMessage();
		if (errorMessage != null && !errorMessage.isEmpty()) {
			Msg.warn(this, "Error decompiling " + function.getName() + ": " + errorMessage);
			writer.write("// decompile error: ");
			writer.write(errorMessage);
			writer.write('\n');
			return;
		}

		DecompiledFunction decompiledFunction = results.getDecompiledFunction();
		if (decompiledFunction == null) {
			writer.write("// decompile error: empty result\n");
			return;
		}

		String code = decompiledFunction.getC();
		if (code != null) {
			writer.write(code);
		}
	}
}
