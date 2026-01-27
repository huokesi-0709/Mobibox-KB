import sqlite3
import sqlite_vec

def main():
    c = sqlite3.connect(":memory:")
    c.enable_load_extension(True)
    sqlite_vec.load(c)
    c.enable_load_extension(False)

    # 查看 vec_* 函数是否存在（只是打印，不影响后续）
    rows = c.execute("select name from pragma_function_list where name like 'vec_%' order by name").fetchall()
    print("vec functions:", [r[0] for r in rows][:30], " ... total=", len(rows))

    # 创建 vec0 表并插入一个 3维向量（这里先不测插入，插入放在主工程里测）
    c.execute("create virtual table v using vec0(embedding float[3])")
    print("vec0 table created OK")
    c.close()

if __name__ == "__main__":
    main()