#include <TFile.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPaveText.h>

void make_plot() {
    // Define your variables here
    const char* selection = "selNone";
    const char* variable = "RecoDiJet_delta_R";
    const bool normalisation = true;
    const double luminosity = 10000;
    const bool log_scale = true;
    const char* input_dir = "/eos/user/t/tcritchl/outputs/output_final/testingAll/";
    const char* output_dir = Form("Background/%s/", selection);

    // Create the canvas
    TCanvas* c = new TCanvas("can", "can", 800, 600);

    // Load your files and create histograms
    TFile* file_Zud = TFile::Open(Form("%sp8_ee_Zud_ecm91_%s_histo.root", input_dir, selection));
    TFile* file_Zcc = TFile::Open(Form("%sp8_ee_Zcc_ecm91_%s_histo.root", input_dir, selection));
    TFile* file_Zbb = TFile::Open(Form("%sp8_ee_Zbb_ecm91_%s_histo.root", input_dir, selection));

    TH1F* h_Zud = (TH1F*)file_Zud->Get(variable);
    TH1F* h_Zcc = (TH1F*)file_Zcc->Get(variable);
    TH1F* h_Zbb = (TH1F*)file_Zbb->Get(variable);

    if (normalisation) {
        double cross_sections[] = { 11870.5, 5215.46, 6654.46 };
        double total_events[] = { 497.658684, 499.786495, 438.738637 };

        double scaling_factor_Zud = (cross_sections[0] * luminosity) / total_events[0];
        double scaling_factor_Zcc = (cross_sections[1] * luminosity) / total_events[1];
        double scaling_factor_Zbb = (cross_sections[2] * luminosity) / total_events[2];

        h_Zud->Scale(scaling_factor_Zud);
        h_Zcc->Scale(scaling_factor_Zcc);
        h_Zbb->Scale(scaling_factor_Zbb);
    }

    h_Zud->SetLineColor(856);
    h_Zcc->SetLineColor(410);
    h_Zbb->SetLineColor(801);

    h_Zud->SetLineWidth(3);
    h_Zcc->SetLineWidth(3);
    h_Zbb->SetLineWidth(3);

    // Create and configure the legend
    TLegend* leg_bg = new TLegend(0.15, 0.75, 0.35, 0.90);
    leg_bg->SetFillStyle(0);
    leg_bg->SetLineWidth(0);
    leg_bg->AddEntry(h_Zud, "Z -> ud", "l");
    leg_bg->AddEntry(h_Zcc, "Z -> cc", "l");
    leg_bg->AddEntry(h_Zbb, "Z -> bb", "l");

    // Create and configure the title
    TPaveText* title = new TPaveText(0.15, 0.85, 0.30, 0.90, "NDC");
    title->AddText("FCCee Simulation");
    title->SetBorderSize(0);
    title->SetFillStyle(0);

    // Draw the histograms and legend
    h_Zud->Draw("hist");
    h_Zcc->Draw("hist same");
    h_Zbb->Draw("hist same");
    leg_bg->Draw();
    title->Draw();

    // Set log scale if necessary
    if (log_scale) {
        c->SetLogy();
    }

    // Save the plot as a PDF
    c->SaveAs(Form("%sBackground_%s%s.pdf", output_dir, selection, variable));
}
