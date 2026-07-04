/* Parallel Ghidra postScript for LLM4Decompile (consumer-stream v2).
 *
 * Decompiles all functions via ParallelDecompiler consumer API, writes per-thread
 * shard files without blocking workers, then merges by address into one C file
 * compatible with service/app/preprocessor.py (// Function: markers).
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import generic.cache.CachingPool;
import generic.cache.CountingBasicFactory;
import generic.concurrent.QCallback;
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.decompiler.DecompiledFunction;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.parallel.ParallelDecompiler;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.CodeUnit;
import ghidra.util.Msg;
import ghidra.util.exception.CancelledException;
import ghidra.util.task.TaskMonitor;

public class DecompileParallel extends GhidraScript {

	private static final int DEFAULT_CHUNK_SIZE = 2000;
	private static final int DEFAULT_SINGLE_QUEUE_LIMIT = 12000;
	private static final String RECORD_ADDR_PREFIX = "@@ADDR@@";
	private static final String RECORD_END = "@@END@@";

	private static class FunctionResult implements Comparable<FunctionResult> {
		private final String marker;
		private final Address address;
		private final String body;

		FunctionResult(String marker, Address address, String body) {
			this.marker = marker;
			this.address = address;
			this.body = body;
		}

		@Override
		public int compareTo(FunctionResult other) {
			return address.compareTo(other.address);
		}
	}

	private static class ShardMergeEntry implements Comparable<ShardMergeEntry> {
		private final FunctionResult result;
		private final BufferedReader reader;

		ShardMergeEntry(FunctionResult result, BufferedReader reader) {
			this.result = result;
			this.reader = reader;
		}

		@Override
		public int compareTo(ShardMergeEntry other) {
			return result.compareTo(other.result);
		}
	}

	private class DecompilerFactory extends CountingBasicFactory<DecompInterface> {
		private final DecompileOptions options;

		DecompilerFactory(DecompileOptions options) {
			this.options = options;
		}

		@Override
		public DecompInterface doCreate(int itemNumber) throws IOException {
			DecompInterface decompiler = new DecompInterface();
			decompiler.setOptions(options);
			decompiler.openProgram(currentProgram);
			decompiler.toggleSyntaxTree(false);
			return decompiler;
		}

		@Override
		public void doDispose(DecompInterface decompiler) {
			decompiler.dispose();
		}
	}

	private class ParallelDecompilerCallback implements QCallback<Function, FunctionResult> {
		private final CachingPool<DecompInterface> pool;
		private final DecompileOptions options;

		ParallelDecompilerCallback(CachingPool<DecompInterface> pool, DecompileOptions options) {
			this.pool = pool;
			this.options = options;
		}

		@Override
		public FunctionResult process(Function function, TaskMonitor monitor) throws Exception {
			if (monitor.isCancelled()) {
				return null;
			}

			DecompInterface decompiler = pool.get();
			try {
				return decompileOne(function, decompiler, monitor);
			}
			finally {
				pool.release(decompiler);
			}
		}

		private FunctionResult decompileOne(Function function, DecompInterface decompiler, TaskMonitor monitor) {
			Address entryPoint = function.getEntryPoint();
			String marker = "// Function: " + function.toString();

			CodeUnit codeUnitAt = function.getProgram().getListing().getCodeUnitAt(entryPoint);
			if (codeUnitAt == null || !(codeUnitAt instanceof Instruction)) {
				String prototype = function.getPrototypeString(false, false) + ';';
				return new FunctionResult(marker, entryPoint, prototype);
			}

			monitor.setMessage("Decompiling " + function.getName());
			DecompileResults results = decompiler.decompileFunction(function, options.getDefaultTimeout(), monitor);
			String errorMessage = results.getErrorMessage();
			if (errorMessage != null && !errorMessage.isEmpty()) {
				Msg.warn(this, "Error decompiling " + function.getName() + ": " + errorMessage);
				return new FunctionResult(marker, entryPoint, "// decompile error: " + errorMessage + "\n");
			}

			DecompiledFunction decompiledFunction = results.getDecompiledFunction();
			if (decompiledFunction == null) {
				return new FunctionResult(marker, entryPoint, "// decompile error: empty result\n");
			}

			String code = decompiledFunction.getC();
			if (code == null) {
				code = "";
			}
			return new FunctionResult(marker, entryPoint, code);
		}
	}

	private static class ShardWriter {
		private final Path shardDir;
		private final ConcurrentHashMap<Long, PrintWriter> writers = new ConcurrentHashMap<>();
		private final AtomicInteger resultCount = new AtomicInteger();

		ShardWriter(Path shardDir) {
			this.shardDir = shardDir;
		}

		void write(FunctionResult result) throws IOException {
			long threadId = Thread.currentThread().getId();
			PrintWriter writer = writers.computeIfAbsent(threadId, id -> {
				try {
					Path shardPath = shardDir.resolve("shard-" + id + ".tmp");
					return new PrintWriter(new FileWriter(shardPath.toFile(), true));
				}
				catch (IOException e) {
					throw new RuntimeException(e);
				}
			});

			synchronized (writer) {
				writer.println(RECORD_ADDR_PREFIX + result.address.toString());
				writer.println(result.marker);
				writer.print(result.body);
				if (!result.body.endsWith("\n")) {
					writer.println();
				}
				writer.println(RECORD_END);
			}
			resultCount.incrementAndGet();
		}

		int getResultCount() {
			return resultCount.get();
		}

		int getShardCount() {
			return writers.size();
		}

		void closeAll() {
			for (PrintWriter writer : writers.values()) {
				writer.close();
			}
		}
	}

	@Override
	protected void run() throws Exception {
		String[] args = getScriptArgs();
		if (args.length < 1 || args[0].trim().isEmpty()) {
			throw new IllegalArgumentException(
				"Usage: DecompileParallel.java <output_path> [segment_size] [max_cpu] [single_queue_limit]");
		}

		String outputPath = args[0].trim();
		int segmentSize = DEFAULT_CHUNK_SIZE;
		if (args.length >= 2 && !args[1].trim().isEmpty()) {
			segmentSize = Integer.parseInt(args[1].trim());
			if (segmentSize < 1) {
				segmentSize = DEFAULT_CHUNK_SIZE;
			}
		}
		int maxCpu = 0;
		if (args.length >= 3 && !args[2].trim().isEmpty()) {
			maxCpu = Integer.parseInt(args[2].trim());
		}
		int singleQueueLimit = DEFAULT_SINGLE_QUEUE_LIMIT;
		if (args.length >= 4 && !args[3].trim().isEmpty()) {
			singleQueueLimit = Integer.parseInt(args[3].trim());
			if (singleQueueLimit < 1) {
				singleQueueLimit = DEFAULT_SINGLE_QUEUE_LIMIT;
			}
		}

		List<Function> allFunctions = new ArrayList<>();
		FunctionIterator iterator = currentProgram.getFunctionManager().getFunctions(true);
		while (iterator.hasNext()) {
			allFunctions.add(iterator.next());
		}

		int totalFunctions = allFunctions.size();
		monitor.initialize(totalFunctions);

		DecompileOptions options = new DecompileOptions();
		CachingPool<DecompInterface> decompilerPool = new CachingPool<>(new DecompilerFactory(options));
		ParallelDecompilerCallback callback = new ParallelDecompilerCallback(decompilerPool, options);

		Path shardDir = Files.createTempDirectory("decompile_parallel_shards");
		ShardWriter shardWriter = new ShardWriter(shardDir);
		String mode = totalFunctions <= singleQueueLimit ? "consumer_stream" : "consumer_segmented";
		long heapMb = Runtime.getRuntime().maxMemory() / (1024L * 1024L);

		long decompileStartMs = System.currentTimeMillis();
		try {
			Consumer<FunctionResult> resultConsumer = result -> {
				if (result == null) {
					return;
				}
				try {
					shardWriter.write(result);
				}
				catch (IOException e) {
					throw new RuntimeException(e);
				}
			};

			if (totalFunctions <= singleQueueLimit) {
				ParallelDecompiler.decompileFunctions(
					callback,
					currentProgram,
					allFunctions.iterator(),
					resultConsumer,
					monitor
				);
			}
			else {
				for (int start = 0; start < allFunctions.size(); start += segmentSize) {
					monitor.checkCanceled();
					int end = Math.min(start + segmentSize, allFunctions.size());
					List<Function> segment = allFunctions.subList(start, end);
					ParallelDecompiler.decompileFunctions(
						callback,
						currentProgram,
						segment.iterator(),
						resultConsumer,
						monitor
					);
				}
			}
		}
		finally {
			shardWriter.closeAll();
			decompilerPool.dispose();
		}
		long decompileMs = System.currentTimeMillis() - decompileStartMs;

		long mergeStartMs = System.currentTimeMillis();
		try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
			mergeShards(shardDir, writer, monitor);
		}
		long mergeMs = System.currentTimeMillis() - mergeStartMs;

		cleanupShardDir(shardDir);

		println(
			"DECOMPILE_PARALLEL mode=" + mode
				+ " functions=" + totalFunctions
				+ " threads=" + maxCpu
				+ " heap=" + heapMb + "M"
		);
		println(
			"DECOMPILE_PARALLEL timing_ms decompile=" + decompileMs
				+ " merge=" + mergeMs
		);
		println(
			"DECOMPILE_PARALLEL shards=" + shardWriter.getShardCount()
				+ " results=" + shardWriter.getResultCount()
		);
	}

	private void mergeShards(Path shardDir, PrintWriter output, TaskMonitor monitor) throws CancelledException, IOException {
		File[] shardFiles = shardDir.toFile().listFiles((dir, name) -> name.startsWith("shard-") && name.endsWith(".tmp"));
		if (shardFiles == null || shardFiles.length == 0) {
			return;
		}

		PriorityQueue<ShardMergeEntry> heap = new PriorityQueue<>();
		List<BufferedReader> openReaders = new ArrayList<>();

		try {
			for (File shardFile : shardFiles) {
				monitor.checkCanceled();
				BufferedReader reader = new BufferedReader(new FileReader(shardFile));
				FunctionResult first = readNextRecord(reader);
				if (first != null) {
					openReaders.add(reader);
					heap.offer(new ShardMergeEntry(first, reader));
				}
				else {
					reader.close();
				}
			}

			while (!heap.isEmpty()) {
				monitor.checkCanceled();
				ShardMergeEntry entry = heap.poll();
				output.write(entry.result.marker);
				output.write(entry.result.body);

				FunctionResult next = readNextRecord(entry.reader);
				if (next != null) {
					heap.offer(new ShardMergeEntry(next, entry.reader));
				}
			}
			output.flush();
		}
		finally {
			for (BufferedReader reader : openReaders) {
				try {
					reader.close();
				}
				catch (IOException e) {
					// Best-effort cleanup after merge.
				}
			}
		}
	}

	private FunctionResult readNextRecord(BufferedReader reader) throws IOException {
		String line;
		while ((line = reader.readLine()) != null) {
			if (!line.startsWith(RECORD_ADDR_PREFIX)) {
				continue;
			}

			String addressText = line.substring(RECORD_ADDR_PREFIX.length()).trim();
			Address address = currentProgram.getAddressFactory().getAddress(addressText);
			if (address == null) {
				continue;
			}

			String markerLine = reader.readLine();
			if (markerLine == null) {
				return null;
			}

			StringBuilder bodyBuilder = new StringBuilder();
			while ((line = reader.readLine()) != null) {
				if (RECORD_END.equals(line)) {
					break;
				}
				bodyBuilder.append(line).append('\n');
			}

			return new FunctionResult(markerLine, address, bodyBuilder.toString());
		}
		return null;
	}

	private void cleanupShardDir(Path shardDir) {
		File[] files = shardDir.toFile().listFiles();
		if (files != null) {
			for (File file : files) {
				file.delete();
			}
		}
		shardDir.toFile().delete();
	}
}
