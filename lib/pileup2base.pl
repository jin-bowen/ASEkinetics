#!/usr/bin/perl
#############################################
#Author: Jiang Li
#email: riverlee2008@gmail.com
#Creat Time: Thu 15 Sep 2011 08:41:43 AM CDT 
#Vanderbilt Center for Quantitative Sciences
#############################################
use strict;
use warnings;
use File::Basename;
#Usage: perl pileup2base.pl pileupfile BQcutoff outputfile
my $usage = <<USAGE;
Usage: perl pileup2base.pl pileupfile BQcutoff outputfile
USAGE

if(scalar(@ARGV) != 3){
    print $usage;
    exit(0);
}

my ($input,$BQcut,$output) = @ARGV;
if(! -e $input){
    die "Input file '$input' does not exists\n";
    exit(1);
}

if($BQcut!~/^\d/){
    die "Base quality cutoff (BQcut) is not numeric\n";
    exit(1);
}

#Do the parsing
open FILE, $input or die "error, can not open $input";
open WFILE, '>', $output or die "error can not open $output to write";
print WFILE "chr\t"."loc\t"."ref\t"."A\t"."T\t"."C\t"."G\n";
print "[",scalar(localtime),"] Begin parsing...\n";
while(<FILE>){
    s/\r|\n//g;
    my($chr,$loc,$ref,$dp,$bases,$bq,$SRA,$CB,$UB,$GENE) = split /\s+/;
    $ref=uc($ref);
   
    my @CBS=split(',', $CB);
    my @UBS=split(',', $UB);
    my @GENES=split(',', $GENE);
 
    next if($dp<1);
    #do some modificaton on $base to remove additional characters
    #1,remove the ^. pattern
    $bases=~s/\^.//g;
    #2,remove the $ pattern
    $bases=~s/\$//g;
    #3,remove -[0-9]+[ACGTNacgtn]+ pattern
    my %hash=();
    while($bases=~/-(\d+)/g){
        $hash{$1}=1;
    }
    foreach my $k (keys %hash){
        $bases=~s/-$k[ACGTNacgtn]{$k}//g;
    }
    %hash=();
    while($bases=~/\+(\d+)/g){
        $hash{$1}=1;
    }
    foreach my $k (keys %hash){
        $bases=~s/\+$k[ACGTNacgtn]{$k}//g;
    }
    
    #Now @base and @bq have the same length
    my @base=split(//,$bases);
    my @bq=split(//,$bq);
    #I have check it
    #if(scalar(@base) ne scalar(@bq)){
    #	print $_,"\n";
    #}
    #foreach my $c (@base){
    #	$check{$c}++;
    #}
    my $A="";
    my $T="";
    my $C="";
    my $G="";
    my $a="";
    my $t="";
    my $c="";
    my $g="";

    #start the loop
    for(my $i=0;$i<@base;$i++){
        my $ch=$base[$i];
        my $score=ord($bq[$i])-33;
	my $record=$CBS[$i].":".$UBS[$i].":".$GENES[$i].",";

	next if($record =~ /\*/);

        if($score>=$BQcut){
            if($ch eq "A"){
                $A .=$record;
            }elsif($ch eq "T"){
                $T .=$record;
            }elsif($ch eq "C"){
                $C .=$record;
            }elsif($ch eq "G"){
                $G .=$record;
            }elsif($ch eq "a"){
                $a .=$record;
            }elsif($ch eq "t"){
                $t .=$record;
            }elsif($ch eq "c"){
                $c .=$record;
            }elsif($ch eq "g"){
                $g .=$record;
            }elsif($ch eq "."){
                if($ref eq "A"){
                    $A .=$record;
                }elsif($ref eq "T"){
                    $T .=$record;
                }elsif($ref eq "C"){
                    $C .=$record;
                }elsif($ref eq "G"){
                    $G .=$record;
                }
            }elsif($ch eq ","){
                if($ref eq "A"){
                    $a .=$record;
                }elsif($ref eq "T"){
                    $t .=$record
                }elsif($ref eq "C"){
                    $c .=$record;
                }elsif($ref eq "G"){
                    $g .=$record;
                }
            }
        }#end the condition  $score>=$BQcut
    }#end the loop		

    $A=$A.$a;
    $T=$T.$t;
    $C=$C.$c;
    $G=$G.$g;
    chop($A);
    chop($T);
    chop($C);
    chop($G);
    print WFILE "$chr"."\t"."$loc"."\t".$ref."\t".$A."\t".$T."\t".$C."\t".$G."\n";	
}#end the reading of the file
close FILE;
close WFILE;

print "[",scalar(localtime),"] Finished\n";
