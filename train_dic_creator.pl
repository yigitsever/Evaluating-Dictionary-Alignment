#!/usr/bin/env perl
#
#
#Copyright © 2019 Yiğit Sever <yigit.sever@tedu.edu.tr>

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# Get source language code and target language code
# optionally give cutoff, cutoff/2 pairs will be prepared for train/test
# optionally give a different dictionary directory name
#
# USAGE:
# $ perl train_dic_creator.pl <source_lang> <target_lang> (cutoff) (dictionary_dir)

use strict;
use warnings;
use List::Util qw(shuffle);

my ($source_lang, $target_lang, $cutoff, $dict_dir) = @ARGV;

if (not defined $source_lang or not defined $target_lang) {
    die "usage: ./train_dic_creator.pl <source_lang> <target_lang> (cutoff)";
}

if (not defined $cutoff && $cutoff ne '') {
    $cutoff = 20000;
}

if (not defined $dict_dir && $dict_dir ne '') {
    $dict_dir = './dictionaries/';
}

my $flipped = 0;
my $file_name;

if (-e "$dict_dir/$target_lang-$source_lang.dic") {
    warn "Dictionary is formatted as $target_lang $source_lang, still creating $source_lang $target_lang";
    $file_name = "$target_lang-$source_lang.dic";
    $flipped = 1;
} elsif (-e "$dict_dir/$source_lang-$target_lang.dic") {
    $file_name = "$source_lang-$target_lang.dic";
}

my $file_path = $dict_dir . $file_name;

local @ARGV = $file_path;
local $^I = '.bak';

while (<>) { # remove empty lines
    print if ! /^$/;
}

my @lines = `sort -rn $file_path`; # better translations swim to top

my @result;
my $c = 0;

foreach my $line (@lines) {
    chomp($line);
    if ($line !~ m/^\d+\s+[0-9.]+\s+(\S+)\s+(\S+)\s+[0-9.]+\s+[0-9.]+$/) {
        # line has multiple tokens
        next;
    } else {
        my ($source, $target) = $line =~ m/^\d+\s+[0-9.]+\s+(\S+)\s+(\S+)\s+[0-9.]+\s+[0-9.]+$/;

        if ($flipped) { # The file name and given parameters mismatch, correcting
            push @result, "$target $source";
        } else {
            push @result, "$source $target";
        }
        $c++;

        if ($c >= $cutoff) {
            last;
        }
    }
}

my $test = scalar @result;

if ($cutoff > scalar @result) {
    $cutoff = scalar @result;
}

@result = shuffle @result;

my $size = $cutoff / 2;

my @head = @result[0..$size - 1];
my @tail = @result[-$size..-1];

my $train_file_name = $source_lang . '_' . $target_lang . '.train';
my $test_file_name = $source_lang . '_' . $target_lang . '.test';

open my $train_fh, '>', $train_file_name;
open my $test_fh, '>', $test_file_name;

print $train_fh join("\n", @head);
print $test_fh join("\n", @tail);

unlink "$file_path$^I";
