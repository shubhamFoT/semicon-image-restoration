`timescale 1ns / 1ps

/*
 * Module: ssim_diff_squarer
 * Description: A 2-stage pipelined hardware accelerator for SSIM computation.
 * It computes the squared difference between two pixels (or local means), 
 * which is the core bottleneck for variance and covariance calculations in SSIM.
 */
 
module ssim_diff_squarer #(
    parameter DATA_WIDTH = 8 // Standard 8-bit grayscale pixel
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [DATA_WIDTH-1:0] pixel_a,
    input wire [DATA_WIDTH-1:0] pixel_b,
    output reg [(2*DATA_WIDTH)-1:0] squared_diff,
    output reg valid_out
);

    // --- Pipeline Stage 1 Registers ---
    reg signed [DATA_WIDTH:0] diff_reg; 
    reg valid_stage1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            diff_reg <= 0;
            valid_stage1 <= 1'b0;
            squared_diff <= 0;
            valid_out <= 1'b0;
        end else begin
            
            // --- STAGE 1: Compute Signed Difference ---
            if (valid_in) begin
                diff_reg <= $signed({1'b0, pixel_a}) - $signed({1'b0, pixel_b});
            end
            valid_stage1 <= valid_in;

            // --- STAGE 2: Compute Square ---
            if (valid_stage1) begin
                squared_diff <= diff_reg * diff_reg;
            end
            valid_out <= valid_stage1;
            
        end
    end

endmodule
