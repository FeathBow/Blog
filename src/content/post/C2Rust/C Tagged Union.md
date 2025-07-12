---
title: C Tagged Union
description: 
tags: []
publishDate: 2025-07-12 03:00
draft: true
share: true
---

# C Tagged Union

C 并不直接支持带标签联合体这一原生特性，但通过**约定**和**组合使用 struct/union/enum**等，可以实现类似的**变体类型**。
## 结构体内联 Tag + Union

这是**最典型也最推荐**的模式：结构体里包含一个表明当前类型的 tag 字段，以及一个匿名或命名的 union 存储变体数据。Tag 通常是枚举或整数常量，union 则列出所有可能的数据类型。例如：

```C title="Tag + Union"
enum ExprType { 
    EXPR_NUM,
    EXPR_NEG,
    EXPR_ADD,
    EXPR_MUL
};
struct Expr {
    enum ExprType type;
    union {
        int num;
        struct Expr* neg;
        struct {
            struct Expr *l, *r;
        } bin;
    } data;
};
```

在这个结构里，`type` 和 `data` 字段紧密关联：程序应根据 `type` 的值判断 `data` 联合体目前存储的是哪一种数据。访问时典型用法如：

```C title="Usage"
switch (expr->type) {
case EXPR_NUM:
    printf("%d", expr->data.num);
    break;
case EXPR_NEG:
    eval(expr->data.neg);
    break;
case EXPR_ADD:
    return eval(expr->data.bin.l) + eval(expr->data.bin.r);
    …
}
```

这一模式**广泛出现**于需要表示“多种类型共享操作”的场景，如编译器/解释器的 AST 节点、UI 事件结构（不同事件类型，附带不同数据）、内核的系统调用参数等。使用这个模式时最好将 tag 定义为 `enum` 而非裸的 int，以利用编译器检查（例如开启 `-Wswitch-enum` 警告可以提醒遗漏的分支）很多教程 [^1] 也常以这个模式作为 C 联合体的高级用法示例。

:::note[优点] Tag 和 union 同处一 struct，使用方便，逻辑清晰。易于通过函数封装实现安全访问，例如封装一个 `get_num()` 函数，在其中校验 `type == EXPR_NUM` 再返回值。缺点是需要程序员自觉维持二者一致，修改 tag 或 union 时需同步更新对应定义，否则可能出现未定义行为。:::

## 分离的 Tag（全局或外部）

有些代码未将 tag 和 union 放在同一结构体内，而是**将 tag 放在其他地方**（可能是全局变量、同一函数的不同变量、甚至隐藏在别的结构中）。例如：

```C title="Global Tag"
int current_kind;
union Data {
    int i;
    float f;
    double d;
} current_val;
```

这里用全局变量 `current_kind` 表示 `current_val` 当前存放的数据类型。
```C title="Usage"
if (current_kind == 0) {
    printf("Int: %d\n", current_val.i);
} else if (current_kind == 1) {
    printf("Float: %f\n", current_val.f);
}
…
```

这种模式在早期代码或特殊场景下出现，即 tag 信息不是紧贴着数据存储，而由上下文约定提供。例如某状态机使用一个全局状态变量代表当前数据解释方式，或者在网络协议处理时根据报文上下文（如协议版本字段）决定如何解析后续联合体。再如一些函数会并行传入一个类型码和一个 `void*` 指针参数，这里的类型码就扮演 tag 的角色，而 `void*` 可以指向不同结构。

:::caution Tag 分离后，没有语言机制强制它和数据关联，因此**更易出错**。如果多个不同模块或线程访问/修改这些变量，维护一致性难度更大。此外，静态分析工具更难跟踪这种模式下 tag 和 union 的关系。论文中提到，有些程序依赖全局上下文决定读取哪个字段而非使用明确的 tag 字段。对于自动转换工具来说，识别这类关联是一个挑战，往往需要用户介入或特别约定，转换为 Rust 后也可能需要重构设计以消除全局可变状态。 :::

## 多重嵌套的联合体

联合体可以**递归或嵌套**出现，形成复杂的数据结构。例如一个 union 的某个成员本身又是一个 struct/union 组合，这就引入了**多级标签**。

```C title="Multi Level"
enum OuterTag {
    OT_INT,
    OT_COMPLEX
};
enum InnerTag {
    IT_A,
    IT_B,
    IT_C
};

struct Outer {
    enum OuterTag tag;
    union {
        int simple;
        struct {
            enum InnerTag sub_tag;
            union {
                long a;
                double b;
                char* c;
            } inner;
        } complex;
    } u;
};
```

在上例中，`Outer` 有两级类型区分：`tag` 决定使用 `simple` 还是 `complex` 分支；如果是 `complex`，则还要看 `sub_tag` 决定 `inner` 联合体用哪种类型（long、double、char*）。访问代码可能像：

**多重嵌套**在一些复杂系统中出现，如编译器前端 AST 可能有多层分类，或者网络协议解析时有分层的包头格式。POSIX 的 `siginfo_t` 结构就是嵌套联合的著名案例，它在一个结构内先区分信号大类，再根据子类型选择不同的数据结构 [^2]。

```C title="siginfo.h"
typedef struct siginfo {
    int si_signo; /* Signal number.  */
    int si_errno; /* If non-zero, an errno value associated with
                    this signal, as defined in <errno.h>.  */
    int si_code; /* Signal code.  */

    union {
        int _pad[__SI_PAD_SIZE];

        /* kill().  */
        struct
        {
            __pid_t si_pid; /* Sending process ID.  */
            __uid_t si_uid; /* Real user ID of sending process.  */
        } _kill;

        /* POSIX.1b timers.  */
        struct
        {
            int si_tid; /* Timer ID.  */
            int si_overrun; /* Overrun count.  */
            sigval_t si_sigval; /* Signal value.  */
        } _timer;

        /* POSIX.1b signals.  */
        struct
        {
            __pid_t si_pid; /* Sending process ID.  */
            __uid_t si_uid; /* Real user ID of sending process.  */
            sigval_t si_sigval; /* Signal value.  */
        } _rt;

        /* SIGCHLD.  */
        struct
        {
            __pid_t si_pid; /* Which child.  */
            __uid_t si_uid; /* Real user ID of sending process.  */
            int si_status; /* Exit value or signal.  */
            __clock_t si_utime;
            __clock_t si_stime;
        } _sigchld;

        /* SIGILL, SIGFPE, SIGSEGV, SIGBUS.  */
        struct
        {
            void* si_addr; /* Faulting insn/memory ref.  */
        } _sigfault;

        /* SIGPOLL.  */
        struct
        {
            long int si_band; /* Band event for SIGPOLL.  */
            int si_fd;
        } _sigpoll;
    } _sifields;
} siginfo_t;
```

其中 `si_code` 是主 tag，不同值对应 `_sifields` 联合中的不同子结构（`_kill`, `_timer`, …），而某些子结构内又含有进一步的细分字段。为了方便使用，glibc 通过宏把各分支字段映射为统一的名称（如 `si_pid` 在不同 union 成员中都有定义）。

:::note 嵌套联合体需要识别多个层级的标签，以及可能存在多个不同层级 tag 之间的关联（如上例 `OuterTag` 和 `InnerTag` 共同决定数据）。一般策略是逐层转换，每一层分别应用前述方法，但要特别注意避免改变原内存布局和对齐。:::
## 匿名联合

C11 开始标准支持**匿名联合**，某些编译器在此之前就提供了类似扩展。匿名联合的特点是**它没有名字，直接将成员暴露给包含它的作用域**。

```C title="Anonymous Union"
struct Event {
    int type;
    union { 
        int i;
        float f;
    };
};
```

在这个结构里，可以直接用 `evt.i` 或 `evt.f` 访问 union 成员，而不需要 `evt.u.i` 这样的中间名。Tag 字段 `type` 仍然存在，但联合体没有额外的标识符。这种写法只是语法上的简化，语义上等价于命名一个比如 `union { int i; float f; } value;` 然后访问时 `evt.value.i`。

:::note 需要注意的是，匿名联合有时也以**匿名 struct + 联合**结合出现，用于实现**共用内存的不同视图** :::

```C title="Anonymous Union"
typedef struct {
    union {
        struct {
            BYTE R, G, B, A;
        } components;
        uint32_t value;
    };
} COLOR;
```

这意味着 `COLOR` 既可按 `.components.R` 等访问字节，又可按 `.value` 访问整体。这虽然不涉及 tag，但属于联合体的一种使用形式。Rust 中可以通过 `union` 实现但需要 unsafe，或通过转型实现。这类匿名用法转换成 Rust 通常直接赋予临时名称，加上一些辅助方法（因为 Rust 枚举不适用此无标签情形，可能需要保持为 `union` 或其他逻辑）。

## 一个 Tag 管理多个 Union / 多个 Tag 控制单个 Union

典型情况下，一个 tag 字段对应控制一个 union。但在较为复杂的结构中，也存在**一个 tag 控制多个 union 字段**或者**多个 tag 共同决定一个 union 使用哪种类型**的情况。
1. **单 tag 多 union**：结构体内可能有**多个 union 字段**，但通过一个通用的 tag 加以区分。
	假设 `kind==0` 表示 `u1.s` 有效，`kind==1` 表示 `u1.n` 有效，`kind==2` 表示 `u2.f` 有效，`kind==3` 表示 `u2.d` 有效。这实际上把一个 tag 的值域划分给了两个不同 union（如 0/1 对应第一个 union，2/3 对应第二个）。这在需要节省内存又想把不相关的数据集中到一个结构时可能出现，但会增加代码维护复杂度。
	
	对于 C2Rust 转换而言，需要将这种结构转换为**嵌套枚举**或**更高级的枚举定义**。一种方法是将 `u1` 和 `u2` 折叠为一个大的枚举，其变体涵盖原来的四种情况（例如 Rust enum 有四个变体：`SVariant(char*)`, `NVariant(int)`, `FVariant(float)`, `DVariant(double)`），然后 `Container` 里只有这一个枚举。这种转换需要分析 tag 值范围属于哪个 union，非常考验静态分析精度。如果分析不出，则可能退而求其次，只转换其中一部分或保持原状。
```C title="Single Tag Multi Union"
struct Container {
    int kind;
    union {
        char* s;
        int n;
    } u1;
    union {
        float f;
        double d;
    } u2;
};
```
2. **多 tag 单 union**：可能存在一个 union 的选择依赖**多个不同的标签**。例如一个主状态码 + 次级类型码共同决定 union 用哪种成员。这往往可以通过**分层拆分**来等价处理（即主状态先决定选哪个子 struct，子 struct 内再用次级 tag 决定 union）。转换成 Rust 时，多半可以转为**枚举嵌套**枚举的形式。也有可能两个 tag 是并列的。
	这里 `is_ptr` 和 `is_signed` 两个布尔共同决定 `data` 用哪个字段：如果 `is_ptr==1` 用 `p`，否则根据 `is_signed` 选 `i` 或 `u`。这种情形可以看作 tag 扩展为两位的信息。Rust 转换可将其归约为一个三态的枚举（例如变体：`Uint(u32)`, `Int(i32)`, `Ptr(*mut c_void)`），同时用逻辑保证原先两标志的组合映射到正确变体。自动分析需要推断这种逻辑（有一定难度）。因此实际迁移中，可能将两个 bool 合成为一个小型 enum（比如三种有效值和一个无效组合）或者干脆保持原来结构，加静态断言约束。
```C title="Single Union Multi Tag"
struct Combo {
    bool is_ptr;
    bool is_signed;
    union {
        unsigned int u;
        int i;
        void* p;
    } data;
};
```

总之，这类**多对多关系**的 Tagged Union 使用，在项目中不是主流但也存在。它们通常源于对内存布局或接口兼容性的特殊要求。例如 Linux 内核很多结构为了节省空间，常把不同但不相干的字段放在共用联合里，并用几个标志位组合表示状态。

# 典型应用场景

**带标签的联合体**广泛出现在系统编程领域，尤其是下面这些场景：

- **编译器/解释器**
	用 Tagged Union 表示抽象语法树（AST）节点或字节码指令。这类结构在不同节点类型携带不同信息，例如前述 `Expr` 示例即源自一个表达式树的定义。类似地，小型语言解释器（如 JSON 解析器的值类型、VM 的指令格式）都常用一个 tag+union 来描述当前这个对象是哪种类型以及相关数据。
- **操作系统内核/驱动**：内核需要处理众多不同类型但格式相关的对象。例如：
	- **信号处理**：POSIX 的 `siginfo_t`[^2] 用一个联合涵盖不同信号的详细信息，由 `si_code` 标签识别。
    - **设备 IO**：许多 ioctl 调用接受的参数结构体包含联合，具体哪个字段有效由命令号（其实相当于 tag）决定。
    - **文件系统/网络**：如 Linux 的 `struct sockaddr` 虽不是 union，但在接口上等同于一个带类型码（address family）的变体结构。经典的 BSD Socket API 要求把特定地址结构（IPv4、IPv6 等）强转为 `sockaddr*` 传入，通信双方通过 `sa_family` 判断实际结构 [^3]。虽然没有使用 `union` 关键字，但这种按照 `sa_family` 解释同一内存区域的做法，本质就是一个 Tagged Union 模式（甚至产生了很多转换和对齐问题）。内核内部在新版实现中，已经尝试把 `sockaddr` 用 union + 灵活数组重定义，以更明确表达多种地址类型的共用。
- **网络协议栈**：处理数据包时，不同协议头部有不同格式。例如 IP 分片包头 vs 普通包头，可以定义为一个包含两种格式的联合，根据标志位决定用哪种解析。TLS/SSL 等也有 content type 字段决定后续结构体格式的情况。很多协议实现直接用 `switch` 判断类型码，然后把同一缓冲区按不同 struct 解读——这可以抽象成 C 中的 tagged union，只是数据来源是网络字节流而已。
    
- **安全敏感的数据结构**：有时为了节约空间或接口统一，一些 API 会让用户填一个结构，其中带 tag 和 union。例如 Linux 内核的 `epoll_event`：
    这里没有显式 tag，因为用户可以自行决定用 `data` 存放哪种类型（例如存文件描述符或指针）。这个 union 的“tag”隐含在用户的认知中。很多底层接口（包括 Windows 的许多结构如 `OVERLAPPED` 等）都有类似 union，让用户选择赋予不同类型的数据。**注意**：这种无显式 tag 的 union 用法，如果误解了期望类型，极易造成错误或漏洞（例如 epoll 本来打算用 `data.ptr` 存 pointer，却错当 `data.u64` 读取）。
```C
typedef union epoll_data {
    void* ptr;
    int fd;
    uint32_t u32;
    uint64_t u64;
} epoll_data_t;

struct epoll_event {
    uint32_t events;
    epoll_data_t data;
};
```

[^1]: [Tagged Unions are actually quite sexy](https://ciesie.com/post/tagged_unions/#:~:text=of%20the%20union%E2%80%99s%20variant%20is,by%20creating%20a%20tagged%20union)

[^2]: [siginfo.h](https://sites.uclouvain.be/SystInfo/usr/include/bits/siginfo.h.html#:~:text=struct%20,%7D%20_kill)

[^3]: [The trouble with struct sockaddr's fake flexible array [LWN.net]](https://lwn.net/Articles//997094/#:~:text=kernel%20to%3A)