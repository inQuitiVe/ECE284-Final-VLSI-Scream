# implementation & discussion
1. 目前最大的問題: psum 到底怎麼流動 ??
method1: 有 open source 直接不用流動的 改成從 row/array 蒐集全部 mac 的 psum 成為一個長到爆的值 (fan out 太高不合理)
method2: 作業的 ws 是 act 和 wgt 共用一 w/e 流動, 可以合理推測 os 可以 psum 和 act or wgt 一起流動 (時序上不允許)
method3: 新增額外的 inst - last data, 借用 s 的 port 在最後一個 weight 輸入完的下一個 cycle 丟出當前的 psum, 
            並在 mac_array 那端把同 col 各 PE 的 out_s 先與 inst[2] (flush psum bit) 做 AND mask，
            然後再對所有 row 做 OR (風險是不能在 8 個 cycle 出現兩次 last data - 但不應該會發生)

2. psum 是否需要 preload ??
    目前不 load 白 不 load，畢竟為了跟 WS 合本來 stream line 就需要 psum_bit 的寬度. 仿造 WS 中 wgt 的 preload 法 

3. inst - last data 是否需要從 we 轉為 ns ?
    不需要，兩者等價 都是隔一個 cycle

4. 還是搞不懂 
    a_q_nxt             = (inst_w[0] || inst_w[1]) ? in_w : a_q; 
    為啥不是
    a_q_nxt             = ((inst_w[0] && ~load_ready_q) || inst_w[1]) ? in_w : a_q; 


# task list
1. 將 mac 做成 2-bit/ 4-bit 可切換                          v
2. 將 mac_tile 改成 os                                      v
3. 將 mac_row 改成 os                                       v
4. 將 mac_array 改成 os                                    v
5. 完成 IFIFO
6. 驗證
7. 將 mac_tile 改成 os / ws 可切換                          v
8. 將 mac_row 改成 os / ws 可切換                           v
9. 將 mac_array 改成 os / ws 可切換                         v