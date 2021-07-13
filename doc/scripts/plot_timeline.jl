using OhMyJulia
using DataStructures

struct Event
    label::String
    start::Float64
    duration::Float64
end

struct TimeLine
    label::String
    events::Vector{Event}
end

TimeLine(label) = TimeLine(label, [])

mutable struct Task
    label::String
    timeline::TimeLine
    duration::Float64
    wait_for::Vector{Task}
    notify::Vector{Task}
    est::Float64
end

Task(label, timeline, duration) = Task(label, timeline, duration, [], [], -1)

function waitfor(a::Task, b::Task)
    push!(a.wait_for, b)
    push!(b.notify, a)
end

function gen_svg(timelines::Vector{TimeLine}, line_height=16, title_offset=20)
    function wrap_svg(f, max_width, max_height)
        header = """
            <svg viewBox="-2 -2 $(max_width+title_offset+4) $(max_height+4)" xmlns="http://www.w3.org/2000/svg" version="1.1">
                <style> .t { font: italic 6px sans-serif; } </style>
                <rect x="-2" y="-2" width="$(max_width+title_offset+4)" height="$(max_height+4)" fill="white"/>
        """

        buffer = IOBuffer() << header << f << "</svg>"

        String(take!(buffer))
    end

    function write_line(buffer, vertical_offset, timeline::TimeLine)
        buffer << """
            <text x="0" y="$(vertical_offset+line_height/2)" class="t" alignment-baseline="middle" text-anchor="start">$(timeline.label):</text>
        """

        for event in timeline.events
            buffer << """
                <rect x="$(title_offset + event.start)" y="$vertical_offset" width="$(event.duration)" height="$line_height" stroke="black" fill="transparent" />
                <text x="$(title_offset + event.start + event.duration/2)" y="$(vertical_offset + line_height/2)" class="t" alignment-baseline="middle" text-anchor="middle">$(event.label)</text>
            """
        end
    end

    max_width = maximum( maximum(event.start + event.duration for event in timeline.events ) for timeline in timelines if !isempty(timeline.events) )
    max_height = length(timelines) * line_height

    wrap_svg(max_width, max_height) do buffer
        for (timeline_id, timeline) in enumerate(timelines)
            write_line(buffer, line_height * (timeline_id - 1), timeline)
        end
    end
end

# we assume gradients are applied to the weights before switching out.
function forward_backward(ncards=4, nlayers=10; time_calculation=12, time_pcie=16, time_nvlink=10, time_back_calculation=2*time_calculation)
    pcie_in = TimeLine("P_IN")
    gpus = [ TimeLine("G_$i") for i in 1:ncards ]
    nvlinks = [ TimeLine("N_$i") for i in 1:ncards-1 ]
    pcie_out = TimeLine("P_OUT")

    swap_in_forward_tasks = [Task("L$i", pcie_in, time_pcie) for i in 1:nlayers]
    swap_in_backward_tasks = [Task("L$i", pcie_in, time_pcie) for i in nlayers:-1:1]
    calculation_forward_tasks = [Task("F$i", gpus[j], time_calculation) for j in 1:ncards, i in 1:nlayers]
    calculation_backward_tasks = [Task("B$i", gpus[j], time_back_calculation) for j in 1:ncards, i in nlayers:-1:1]
    transfer_forward_tasks = [Task("L$i", nvlinks[j], time_nvlink) for j in 1:ncards-1, i in 1:nlayers]
    transfer_backward_tasks = [Task("L$i", nvlinks[j], time_nvlink) for j in 1:ncards-1, i in nlayers:-1:1]
    transfer_gradient_tasks = [Task("G$i", nvlinks[j], time_nvlink) for j in 1:ncards-1, i in nlayers:-1:1]
    swap_out_tasks = [Task("L$i", pcie_out, time_pcie) for i in nlayers:-1:1]

    # dependency 1: sequential layers
    for tasks in (swap_in_forward_tasks, swap_in_backward_tasks, calculation_forward_tasks, calculation_backward_tasks, transfer_forward_tasks, transfer_backward_tasks, transfer_gradient_tasks, swap_out_tasks)
        for i in 2:nlayers
            if ndims(tasks) == 2
                for j in 1:car(size(tasks))
                    waitfor(tasks[j, i], tasks[j, i-1])
                end
            else
                waitfor(tasks[i], tasks[i-1])
            end
        end
    end

    # dependency 2: calculation after weights ready
    for i in 1:nlayers
        waitfor(calculation_forward_tasks[1, i], swap_in_forward_tasks[i])
        waitfor(calculation_backward_tasks[1, i], swap_in_backward_tasks[i])

        for j in 2:ncards
            waitfor(calculation_forward_tasks[j, i], transfer_forward_tasks[j-1, i])
            waitfor(calculation_backward_tasks[j, i], transfer_backward_tasks[j-1, i])
        end
    end

    # dependency 3: transfer after weight and gradients ready
    for i in 1:nlayers
        waitfor(transfer_forward_tasks[1, i], swap_in_forward_tasks[i])
        waitfor(transfer_backward_tasks[1, i], swap_in_backward_tasks[i])
        waitfor(transfer_gradient_tasks[1, i], calculation_backward_tasks[1, i])

        for j in 2:ncards-1
            waitfor(transfer_forward_tasks[j, i], transfer_forward_tasks[j-1, i])
            waitfor(transfer_backward_tasks[j, i], transfer_backward_tasks[j-1, i])
            waitfor(transfer_gradient_tasks[j, i], calculation_backward_tasks[j, i])
        end
    end

    # dependency 4: swap out after backward
    for i in 1:nlayers
        waitfor(swap_out_tasks[i], calculation_backward_tasks[end,i])
    end

    # dependency 5: backward after forward
    waitfor(swap_in_backward_tasks[1], swap_in_forward_tasks[end])
    for j in 1:ncards
        waitfor(calculation_backward_tasks[j,1], calculation_forward_tasks[j,end])
    end
    for j in 1:ncards-1
        waitfor(transfer_backward_tasks[j,1], transfer_forward_tasks[j,end])
        waitfor(transfer_gradient_tasks[j,1], transfer_forward_tasks[j,end])
    end

    # dependency 6: don't transfer in before last one is used and transfered out, to ensure memory usage
    waitfor(swap_in_backward_tasks[1], transfer_forward_tasks[1, end-1])
    waitfor(swap_in_backward_tasks[1], calculation_forward_tasks[1, end-1])
    waitfor(swap_in_backward_tasks[2], transfer_forward_tasks[1, end])
    waitfor(swap_in_backward_tasks[2], calculation_forward_tasks[1, end])
    for i in 3:nlayers
        waitfor(swap_in_forward_tasks[i], transfer_forward_tasks[1, i-2])
        waitfor(swap_in_forward_tasks[i], calculation_forward_tasks[1, i-2])
        waitfor(swap_in_backward_tasks[i], transfer_backward_tasks[1, i-2])
        waitfor(swap_in_backward_tasks[i], calculation_backward_tasks[1, i-2])
        # waitfor(swap_in_backward_tasks[i], transfer_gradient_tasks[1, i-2])

        for j in 1:ncards-2
            waitfor(transfer_forward_tasks[j, i], transfer_forward_tasks[j+1, i-2])
            waitfor(transfer_forward_tasks[j, i], calculation_forward_tasks[j+1, i-2])

            waitfor(transfer_backward_tasks[j, i], transfer_backward_tasks[j+1, i-2])
            waitfor(transfer_backward_tasks[j, i], calculation_backward_tasks[j+1, i-2])

            waitfor(transfer_gradient_tasks[j, i], transfer_gradient_tasks[j+1, i-2])
        end

        waitfor(transfer_forward_tasks[ncards-1, i], calculation_forward_tasks[ncards, i-2])

        waitfor(transfer_backward_tasks[ncards-1, i], swap_out_tasks[i-2])
        waitfor(transfer_backward_tasks[ncards-1, i], calculation_backward_tasks[ncards, i-2])

        waitfor(transfer_gradient_tasks[ncards-1, i], calculation_backward_tasks[ncards, i-2])
    end

    # schedule
    ready_queue = [ task for tasks in (swap_in_forward_tasks, swap_in_backward_tasks, calculation_forward_tasks, calculation_backward_tasks, transfer_forward_tasks, transfer_backward_tasks, transfer_gradient_tasks, swap_out_tasks) for task in tasks if isempty(task.wait_for) ]
    ready_queue = BinaryHeap(Base.By(x->x.est), ready_queue)
    while !isempty(ready_queue)
        task = pop!(ready_queue)
        if !isempty(task.timeline.events)
            task.est = max(task.est, task.timeline.events[end].start + task.timeline.events[end].duration)
        end
        push!(task.timeline.events, Event(task.label, task.est, task.duration))
        for child in task.notify
            child.est = max(child.est, task.est + task.duration)
            child.wait_for = [ x for x in child.wait_for if x ≠ task ]
            if isempty(child.wait_for)
                push!(ready_queue, child)
            end
        end
    end

    # order the lists
    result = [pcie_in]
    for i in 1:ncards-1
        push!(result, gpus[i], nvlinks[i])
    end
    push!(result, gpus[end], pcie_out)

    result
end



AS("image/svg+xml", gen_svg(forward_backward(time_calculation=12, time_pcie=16, time_nvlink=10)))

open("pcie_dominant.svg", "w") do f
    f << gen_svg(forward_backward(time_calculation=12, time_pcie=24, time_nvlink=8))
end

open("calculation_dominant.svg", "w") do f
    f << gen_svg(forward_backward(time_calculation=24, time_pcie=16, time_nvlink=10))
end

open("nvlink_dominant.svg", "w") do f
    f << gen_svg(forward_backward(time_calculation=14, time_pcie=16, time_nvlink=24))
end

open("x.svg", "w") do f
    f << gen_svg(forward_backward(time_calculation=12, time_pcie=16, time_nvlink=10))
end
