// Q1.a
int count;
while (!end_of_file) {
    vector<string> line = file.readline();
    String lecture_title;
    int slide_number;
    parse_line(linr, &lecture_title, &slide_number);
    if (lecture_title == "In-Memory Distributed Computing using Spark" && slide_number == 28 )
        count++;
}
if (get_node_id() == 0) {
    ...
} else sendInt(0, count);

// Q1.c
int num_of_works = 0;
int my_id = get_node_id();
int num_of_pages = get_num_pages("spark");
for (int i = 1; i < num_of_pages; i++) {
    if (hashToNode(i) == my_id)
        num_of_works++;
}

String[num_of_pages] comment_array;
while (!end_of_file) {
    String line = file.readline();
    String lecture_title;
    int slide_no;
    String comment;
    parse_line(line, &lecture_title, &slide_no, &comment);
    if (lecture_title == "spark")
        comment_array[slide_no].append(comment)
    }

for (int i = 0; i < num_of_pages; i++) {
    int node_id = hashToNode(i);
    if (node_id != my_id) send_comment(node_id, i, comment_array[i]);
    else writeFile(i, comment_array[i]);
}

for (int i = 0; i < num_nodes; i++) {
    if (i != my_id) {
        while (num_of_works != 0) {
            int slide_no;
            String comment = recv_comment(i, &slide_no);
            writeFile(slide_no, comment);
            num_of_works--;
        }
    }
}

// Q2.b
char cache_line[64];
int cache_line_len = 0;
int row_start = x / 2048;
int row_end = (x + 64) / 2048;
int col_start = (x % 2048) / 8;
int col_end = ((x + 64) % 2048) / 8;
int chip_start = x % 8;
int chip_end = (x + 64) % 8;

for (int i_row = row_start; i_row < row_end; i_row++) {
    for (int i_col = col_start; i_col < col_end; i_col++) {
        char from_dram[8];
        DIMM_READ_FROM_CHIPS(i_row, i_col, from_dram);
        if (i_col == col_start) {
            memcpy(cache_line + cache_line_len, from_dram[chip_start], 8 - chip_start);
            cache_line_len += 8 - chip_start;
        } else if (i_col == col_end - 1) {
            memcpy(cache_line + cache_line_len, from_dram, chip_end);
            cache_line_len += chip_end;
        } else {
            memcpy(cache_line + cache_line_len, from_dram, 8);
            cache_line_len += 8;
        }
    }
}

