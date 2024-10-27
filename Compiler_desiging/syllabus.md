1. **01_setup_compiler_project_in_cpp.md** – How do you set up a C/C++ project for building a basic compiler? Walk through initializing a project using CMake or Makefile.

2. **02_lexical_analysis_cpp.md** – Implement a simple lexical analyzer (tokenizer) in C++ that can recognize keywords, identifiers, and operators from a basic programming language.

3. **03_handling_tokens_in_cpp.md** – Write a C++ struct to represent tokens and design a function to generate a list of tokens from a source code input.

4. **04_tokenizer_with_flex.md** – Use `Flex` to generate a lexical analyzer in C/C++ and compare it with a manually implemented lexer. How does `Flex` make the process easier?

5. **05_syntax_parser_in_cpp.md** – Write a recursive descent parser in C++ for parsing arithmetic expressions, supporting basic operators like `+`, `-`, `*`, `/`.

6. **06_syntax_parsing_with_bison.md** – Use `Bison` to generate a parser for a simple language in C/C++. Discuss the advantages of using a parser generator over a manually written parser.

7. **07_abstract_syntax_tree_in_cpp.md** – Design a C++ class structure for an Abstract Syntax Tree (AST). Parse a simple expression and print the resulting AST.

8. **08_handling_expression_ast_in_cpp.md** – Implement traversal of an AST in C++ to evaluate arithmetic expressions. How can tree traversal be used for code interpretation?

9. **09_cpp_symbol_table.md** – Implement a symbol table in C++ using a `std::unordered_map` for storing variable names and their types/values.

10. **10_semantic_analysis_cpp.md** – Write a C++ function to perform semantic checks on an AST, such as type checking and variable declarations.

11. **11_intermediate_code_generation_cpp.md** – Generate intermediate code (such as three-address code) from the AST in C++. Use structs or classes to represent the intermediate instructions.

12. **12_intermediate_representation_for_loops.md** – Implement an intermediate representation for loops in C++, like for-loops or while-loops. How do you manage loop control flow in intermediate code?

13. **13_basic_blocks_in_cpp.md** – Write a C++ function that divides intermediate code into basic blocks. What data structures can you use to represent the blocks?

14. **14_control_flow_graph_in_cpp.md** – Build a control flow graph (CFG) from the basic blocks in C++. Represent the graph using adjacency lists or matrices.

15. **15_code_optimization_in_cpp.md** – Implement basic optimization techniques like constant folding and strength reduction in your intermediate code in C++.

16. **16_register_allocation_in_cpp.md** – Implement a simple register allocation algorithm in C++. How can you map variables to machine registers effectively?

17. **17_code_generation_for_x86_in_cpp.md** – Write a code generation function in C++ that outputs assembly code for a subset of the x86 architecture based on the intermediate code.

18. **18_code_generation_for_arm_in_cpp.md** – Implement code generation in C++ that targets ARM assembly. What are the key differences between x86 and ARM code generation?

19. **19_risc_code_generation_in_cpp.md** – Generate assembly code for a simple RISC (Reduced Instruction Set Computer) architecture in C++. How does RISC simplify code generation?

20. **20_peephole_optimization_cpp.md** – Write a peephole optimization function in C++ that scans and optimizes short sequences of generated assembly code.

21. **21_loop_unrolling_in_cpp.md** – Implement loop unrolling in C++ as an optimization technique for loops in the intermediate representation.

22. **22_constant_propagation_in_cpp.md** – Implement constant propagation in C++, an optimization technique that replaces variables with constant values where possible.

23. **23_dead_code_elimination_cpp.md** – Write a C++ function to eliminate dead code (unreachable or unused instructions) from your intermediate code.

24. **24_cpp_parser_with_flex_bison.md** – Combine `Flex` and `Bison` to create a complete lexer and parser in C++ for a small language. What are the integration challenges, and how can they be addressed?

25. **25_handling_syntax_errors_in_cpp.md** – Implement error reporting and recovery mechanisms in your C++ parser. How can you handle syntax errors gracefully during parsing?

26. **26_register_spilling_in_cpp.md** – Implement register spilling in C++ when there are insufficient machine registers available during code generation.

27. **27_cpp_code_generation_with_llvm.md** – Use LLVM as a backend for generating machine code in your C++ compiler. Write C++ code to generate LLVM IR (Intermediate Representation) and compile it to assembly.

28. **28_static_single_assignment_in_cpp.md** – Implement a conversion of intermediate code to Static Single Assignment (SSA) form in C++. How does SSA simplify optimizations?

29. **29_bytecode_interpreter_in_cpp.md** – Write a bytecode interpreter in C++ to execute bytecode instructions generated by your compiler. What design choices are involved in the interpreter?

30. **30_testing_your_compiler_in_cpp.md** – Design and implement a test suite for your compiler in C++ to verify lexical analysis, parsing, code generation, and optimization phases. How can automated testing ensure compiler correctness?